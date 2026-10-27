// Real provider preparation + joined-bias/strided SET_ROWS controls. Not an oracle.
#include "ggml.h"
#include "ggml-backend.h"
#include <cassert>
static bool refuse_fused=false;
static int refused=0;
static bool supports_for_test(ggml_backend_dev_t device,const ggml_tensor * op) {
    if(refuse_fused && op->op==GGML_OP_MUL_MAT && op->src[0]->ne[1]==3072) {++refused;return false;}
    return ggml_backend_dev_supports_op(device,op);
}
#define ggml_backend_dev_supports_op supports_for_test
#include "../decoder.cpp"
#undef ggml_backend_dev_supports_op

struct Fixture {
    struct Value {std::vector<int64_t> shape;std::vector<float> data;};
    std::map<std::string,Value> values;
    std::vector<cpu_encoder_weight> descriptors;
    void add(std::string name,std::initializer_list<int64_t> shape,float value=0) {
        size_t n=1;for(auto x:shape)n*=x;
        values.emplace(std::move(name),Value{shape,std::vector<float>(n,value)});
    }
    Fixture() {
        auto norm=[&](const std::string & p){add(p+".weight",{384},1);add(p+".bias",{384});};
        auto linear=[&](const std::string & p,int in,int out){add(p+".weight",{in,out});add(p+".bias",{out});};
        norm("decoder.layer_norm");
        for(int layer=0;layer<4;++layer) {
            auto p="decoder.layers."+std::to_string(layer);
            for(auto name:{"self_attn_layer_norm","encoder_attn_layer_norm","final_layer_norm"})norm(p+"."+name);
            for(auto a:{"self_attn","encoder_attn"})for(auto proj:{"q_proj","k_proj","v_proj","out_proj"})linear(p+"."+a+"."+proj,384,384);
            linear(p+".fc1",384,1536);linear(p+".fc2",1536,384);
        }
        for(int layer=0;layer<4;++layer)for(int part=0;part<2;++part) {
            auto & w=values.at(cross_projection_name(layer,part,"weight")).data;
            for(size_t j=0;j<w.size();++j)w[j]=float(int((j*7+layer*13+part*5)%41)-20)/2048.f;
            auto & b=values.at(cross_projection_name(layer,part,"bias")).data;
            for(size_t j=0;j<b.size();++j)b[j]=part?float(int(j%7)-3)/128.f:(j%2?0.f:-0.f);
        }
        for(auto & e:values) {
            auto & v=e.second;cpu_encoder_weight d{};d.name=e.first.c_str();d.data=v.data.data();d.bytes=v.data.size()*4;d.rank=int(v.shape.size());
            for(size_t i=0;i<v.shape.size();++i)d.ne[i]=v.shape[i];descriptors.push_back(d);
        }
    }
};
struct Arena {
    std::vector<ggml_fp16_t> buffers[4][4];
    std::array<cpu_decoder_storage_kv,4> kv;
    Arena() {
        for(int i=0;i<4;++i) {
            for(int j=0;j<4;++j)buffers[i][j].assign(size_t(j<2?448:1500)*384+32,0x3555);
            kv[i]={buffers[i][0].data()+16,buffers[i][1].data()+16,buffers[i][2].data()+16,buffers[i][3].data()+16,
                   448*384*2,1500*384*2};
        }
    }
};
static bool ancestor(ggml_tensor * node,ggml_tensor * target) {
    if(node==target)return true;
    for(auto * src:node->src)if(src && ancestor(src,target))return true;
    return false;
}
static void exact_rows() {
    for(auto type:{GGML_TYPE_F16,GGML_TYPE_F32})for(int threads:{1,2,4})for(float sign_bias:{0.f,-0.f,.125f}) {
        auto * backend=ggml_backend_cpu_init();auto * ctx=ggml_init({4u<<20,nullptr,true});auto * graph=ggml_new_graph(ctx);
        auto * raw=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,3072,5);ggml_set_input(raw);ggml_set_output(raw);
        auto * bias=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,384);ggml_set_input(bias);
        auto * positions=ggml_new_tensor_1d(ctx,GGML_TYPE_I64,5);ggml_set_input(positions);
        auto * a=ggml_new_tensor_2d(ctx,type,384,11),*b=ggml_new_tensor_2d(ctx,type,384,11);
        for(auto * t:{a,b}){ggml_set_input(t);ggml_set_output(t);}
        auto * view=ggml_view_2d(ctx,raw,384,5,raw->nb[1],5*384*sizeof(float));
        auto * actual=cross_bias_set_rows(ctx,a,view,bias,positions);
        auto * expected=ggml_set_rows(ctx,b,ggml_add(ctx,view,bias),positions);
        ggml_build_forward_expand(graph,actual);ggml_build_forward_expand(graph,expected);
        auto allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));assert(ggml_gallocr_alloc_graph(allocator,graph));
        assert(actual->data==a->data && actual->view_src==a && actual->src[1]==view && actual->src[2]==bias && actual->src[3]==positions);
        const int64_t indices[5]={7,3,9,0,5};ggml_backend_tensor_set(positions,indices,0,sizeof(indices));
        std::vector<float> biases(384,sign_bias);ggml_backend_tensor_set(bias,biases.data(),0,384*sizeof(float));
        for(int repeat=0;repeat<4;++repeat) {
            std::vector<float> source(3072*5);for(size_t j=0;j<source.size();++j)source[j]=float(int((j*17+repeat*13)%101)-50)/128.f;
            source[5*384]=-0.f;source[5*384+1]=65520.f;source[5*384+2]=0x1p-24f;
            if(repeat==3){source[5*384+3]=INFINITY;source[5*384+4]=std::numeric_limits<float>::quiet_NaN();}
            ggml_backend_tensor_set(raw,source.data(),0,source.size()*sizeof(float));
            std::vector<uint8_t> initial(ggml_nbytes(a),0x35);for(auto * t:{a,b})ggml_backend_tensor_set(t,initial.data(),0,initial.size());
            assert(ggml_graph_compute_with_ctx(ctx,graph,threads)==GGML_STATUS_SUCCESS);
            std::vector<uint8_t> aa(initial.size()),bb(initial.size());ggml_backend_tensor_get(a,aa.data(),0,aa.size());ggml_backend_tensor_get(b,bb.data(),0,bb.size());assert(aa==bb);
            for(int row:{1,2,4,6,8,10})assert(std::memcmp(aa.data()+row*a->nb[1],initial.data()+row*a->nb[1],a->nb[1])==0);
            std::vector<float> unchanged(source.size());ggml_backend_tensor_get(raw,unchanged.data(),0,source.size()*sizeof(float));assert(std::memcmp(source.data(),unchanged.data(),source.size()*sizeof(float))==0);
        }
        ggml_gallocr_free(allocator);ggml_free(ctx);ggml_backend_free(backend);
    }
}
int main() {
    exact_rows();
    Fixture f;cpu_decoder_config config{384,4,6,1536,1500,448,2};
    auto rejected=[&](std::vector<cpu_encoder_weight> d,const char * message) {
        bool caught=false;try{Decoder value(config,true);value.prepare(d.data(),d.size());}
        catch(const std::runtime_error & e){caught=true;assert(std::strstr(e.what(),message));}assert(caught);
    };
    auto d=f.descriptors;d.pop_back();rejected(d,"missing");
    d=f.descriptors;d.push_back(d.front());rejected(d,"duplicate");
    float extra=0;d=f.descriptors;d.push_back({"unexpected.bias",&extra,4,1,{1,0,0}});rejected(d,"unexpected");
    for(int threads:{1,2,4}) {
        Decoder a(config,true),b(config,true);a.self_type=b.self_type=a.cross_type=b.cross_type=GGML_TYPE_F16;
        refuse_fused=false;a.prepare(f.descriptors.data(),f.descriptors.size());assert(a.cross_weight);
        refuse_fused=true;b.prepare(f.descriptors.data(),f.descriptors.size());assert(!b.cross_weight);refuse_fused=false;
        char error[1024];assert(cpu_decoder_set_cross_threads(&a,threads,error,sizeof(error))==0);assert(cpu_decoder_set_cross_threads(&b,threads,error,sizeof(error))==0);
        for(const auto & entry:b.weights)assert(entry.second->data && entry.second->buffer); // no freed-context aliases
        Arena aa,bb;std::vector<float> input(4*384),encoder(1500*384),self_mask(4*448,-INFINITY),cross_mask(4*1500,0),hidden(384);
        int64_t self_pos[4]={0,1,2,3};std::vector<int64_t> cross_pos(1500);for(int i=0;i<1500;++i)cross_pos[i]=1499-i;
        for(int q=0;q<4;++q)for(int k=0;k<=q;++k)self_mask[q*448+k]=0;
        cpu_decoder_storage_call call{4,4,1500,input.data(),input.size()*4,encoder.data(),encoder.size()*4,
            self_pos,sizeof(self_pos),cross_pos.data(),cross_pos.size()*8,self_mask.data(),self_mask.size()*4,
            cross_mask.data(),cross_mask.size()*4,aa.kv.data(),4,hidden.data(),hidden.size()*4};
        auto saved=cross_pos[1];cross_pos[1]=cross_pos[0];bool invalid=false;
        try{a.validate(call);}catch(const std::runtime_error &){invalid=true;}assert(invalid);cross_pos[1]=saved;
        for(int layer=0;layer<4;++layer)for(int plane=0;plane<4;++plane)for(auto bits:aa.buffers[layer][plane])assert(bits==0x3555);
        a.validate(call);Graph ga;ga.build(a,call);call.kv=bb.kv.data();b.validate(call);Graph gb;gb.build(b,call);
        int mm=0,add=0,set=0,custom=0;ggml_tensor * joined=nullptr;
        for(int i=0;i<ggml_graph_n_nodes(ga.cross_graph);++i) {
            auto * t=ggml_graph_node(ga.cross_graph,i);mm+=t->op==GGML_OP_MUL_MAT;add+=t->op==GGML_OP_ADD;set+=t->op==GGML_OP_SET_ROWS;
            if(t->op==GGML_OP_MUL_MAT)joined=t;
            if(t->op==GGML_OP_CUSTOM) {
                ++custom;assert(t->view_src==t->src[0] && t->src[1]->nb[1]==3072*sizeof(float));
                assert(t->src[2] && t->src[3] && ggml_is_contiguous_rows(t->src[1]));
            }
        }
        assert(mm==1 && add==0 && set==0 && custom==8 && joined);
        int cross_flash=0;for(int i=0;i<ggml_graph_n_nodes(ga.body_graph);++i) {
            auto * t=ggml_graph_node(ga.body_graph,i);
            if(t->op==GGML_OP_FLASH_ATTN_EXT && t->src[1]->ne[1]==1500) {assert(ancestor(t,joined));++cross_flash;}
        }assert(cross_flash==4);
        for(int repeat=0;repeat<3;++repeat) {
            for(size_t j=0;j<encoder.size();++j)encoder[j]=float(int((j*11+repeat*17)%101)-50)/128.f;
            call.kv=aa.kv.data();ga.bind(call);assert(ggml_graph_compute(ga.cross_graph,&ga.cross_plan)==GGML_STATUS_SUCCESS);ga.clear();
            call.kv=bb.kv.data();gb.bind(call);assert(ggml_graph_compute(gb.cross_graph,&gb.cross_plan)==GGML_STATUS_SUCCESS);gb.clear();
            for(int i=0;i<4;++i)for(int j=0;j<4;++j) {
                assert(aa.buffers[i][j]==bb.buffers[i][j]);
                for(int n=0;n<16;++n)assert(aa.buffers[i][j][n]==0x3555 && aa.buffers[i][j][aa.buffers[i][j].size()-1-n]==0x3555);
                if(j<2)for(auto bits:aa.buffers[i][j])assert(bits==0x3555);
            }
        }
        std::printf("CROSS_PREFIX threads=%d matmul=%d bias=%d set_rows=%d arena_fused=%zu arena_original=%zu joined_bytes=%zu\n",threads,mm,add,set,
            ggml_gallocr_get_buffer_size(ga.allocator,0),ggml_gallocr_get_buffer_size(gb.allocator,0),ggml_nbytes(joined));
    }
    assert(refused==3);std::puts("DECODER_CROSS_SCATTER_CONTROLS_PASS");
}
