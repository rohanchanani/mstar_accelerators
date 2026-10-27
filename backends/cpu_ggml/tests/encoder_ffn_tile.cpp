// Actual existing GGML operators, buffers, plans and aliases; no new oracle.
#include "ggml.h"
#include "ggml-backend.h"
#include <cassert>
static bool refuse_tiled=false;
static int refused=0;
static bool ffn_supported_for_test(ggml_backend_dev_t device,const ggml_tensor * op) {
    if(refuse_tiled && op->op==GGML_OP_MUL_MAT &&
       (op->src[1]->ne[1]==128||op->src[1]->ne[1]==92) &&
       ((op->src[0]->ne[0]==384&&op->src[0]->ne[1]==1536)||
        (op->src[0]->ne[0]==1536&&op->src[0]->ne[1]==384))) {++refused;return false;}
    return ggml_backend_dev_supports_op(device,op);
}
#define ggml_backend_dev_supports_op ffn_supported_for_test
#include "../encoder.cpp"
#undef ggml_backend_dev_supports_op

static bool accepts(ggml_backend_t backend, ggml_backend_buffer_type_t type,
                    int k, int n, int rows) {
    auto * ctx=ggml_init({3*ggml_tensor_overhead(),nullptr,true});assert(ctx);
    auto * w=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,k,n);
    auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,rows);
    auto * op=ggml_mul_mat(ctx,w,x);
    w->buffer=ggml_backend_buft_alloc_buffer(type,0);
    const bool result=w->buffer&&ggml_backend_supports_op(backend,op);
    if(w->buffer)ggml_backend_buffer_free(w->buffer);
    w->buffer=nullptr;ggml_free(ctx);return result;
}

struct FFNFixture {
    struct Value {std::vector<int64_t> shape;std::vector<float> data;};
    std::map<std::string,Value> values;
    std::vector<cpu_encoder_weight> descriptors;
    void add(const std::string & name,std::initializer_list<int64_t> shape,float value=0) {
        size_t n=1;for(auto x:shape)n*=x;values.emplace(name,Value{shape,std::vector<float>(n,value)});
    }
    FFNFixture() {
        auto norm=[&](const std::string & p){add(p+".weight",{384},1);add(p+".bias",{384});};
        auto linear=[&](const std::string & p,int in,int out){add(p+".weight",{in,out});add(p+".bias",{out});};
        add("encoder.conv1.weight",{3,80,384});add("encoder.conv1.bias",{384});
        add("encoder.conv2.weight",{3,384,384});add("encoder.conv2.bias",{384});
        add("encoder.embed_positions.weight",{384,1500});norm("encoder.layer_norm");
        for(int i=0;i<4;++i) {
            const auto p="encoder.layers."+std::to_string(i);
            norm(p+".self_attn_layer_norm");norm(p+".final_layer_norm");
            for(auto s:{"q_proj","k_proj","v_proj","out_proj"})linear(p+".self_attn."+s,384,384);
            linear(p+".fc1",384,1536);linear(p+".fc2",1536,384);
        }
        for(auto & e:values) {
            auto & v=e.second;cpu_encoder_weight d{};d.name=e.first.c_str();d.data=v.data.data();d.bytes=v.data.size()*4;d.rank=int(v.shape.size());
            for(size_t i=0;i<v.shape.size();++i)d.ne[i]=v.shape[i];descriptors.push_back(d);
        }
    }
};

int main() {
    int executions=0;size_t max_arena=0;
    for(int geometry=0;geometry<2;++geometry) {
        const int width=geometry?384:32,ffn=geometry?1536:64;
        const std::vector<int> lengths=geometry?std::vector<int>{1500}:std::vector<int>{1,127,128,129,257};
        for(int rows:lengths)for(int threads:{1,2,4}) {
            if(geometry&&threads==1)continue;
            cpu_encoder_config config{width,1,1,ffn,80,rows,threads};Encoder e(config,true);
            e.tiled_ffn=true;e.backend=ggml_backend_cpu_init();assert(e.backend);
            e.weights_ctx=ggml_init({1u<<20,nullptr,true});assert(e.weights_ctx);
            e.dense_ctx=ggml_init({1u<<20,nullptr,true});assert(e.dense_ctx);
            e.graph_ctx=ggml_init({4u<<20,nullptr,true});assert(e.graph_ctx);
            auto * input=ggml_new_tensor_2d(e.weights_ctx,GGML_TYPE_F32,width,rows);ggml_set_input(input);
            const std::string prefix="layer";
            for(const auto & item:std::vector<std::pair<std::string,int>>{
                    {".final_layer_norm.weight",width},{".final_layer_norm.bias",width},
                    {".fc1.bias",ffn},{".fc2.bias",width}}) {
                auto * t=ggml_new_tensor_1d(e.weights_ctx,GGML_TYPE_F32,item.second);
                e.weights.emplace(prefix+item.first,t);
            }
            auto * w1=ggml_new_tensor_2d(e.dense_ctx,GGML_TYPE_Q8_0,width,ffn);
            auto * w2=ggml_new_tensor_2d(e.dense_ctx,GGML_TYPE_Q8_0,ffn,width);
            e.weights.emplace(prefix+".fc1.weight",w1);e.weights.emplace(prefix+".fc2.weight",w2);
            auto * dev=ggml_backend_get_device(e.backend);auto * registry=ggml_backend_dev_backend_reg(dev);
            auto extra=reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
                ggml_backend_reg_get_proc_address(registry,"ggml_backend_dev_get_extra_bufts"));
            auto ** choices=extra?extra(dev):nullptr;
            auto selected=mstar_cpu_ggml::first_supported_q8_provider(choices,ggml_backend_buft_name,
                [&](ggml_backend_buffer_type_t candidate) {
                    for(int query:{rows,std::min(128,rows),rows%128?rows%128:128})
                        if(!accepts(e.backend,candidate,width,ffn,query)||
                           !accepts(e.backend,candidate,ffn,width,query))return false;
                    return true;
                });
            assert(selected);std::fprintf(stderr,"FFN_TILE_CONTROL width=%d ffn=%d rows=%d threads=%d provider=%s\n",
                width,ffn,rows,threads,ggml_backend_buft_name(selected));
            e.weights_buffer=ggml_backend_alloc_ctx_tensors(e.weights_ctx,e.backend);assert(e.weights_buffer);
            e.dense_buffer=ggml_backend_alloc_ctx_tensors_from_buft(e.dense_ctx,selected);assert(e.dense_buffer);
            for(const auto & entry:e.weights) {
                auto * t=entry.second;
                if(t->type==GGML_TYPE_Q8_0) {
                    std::vector<float> values(ggml_nelements(t));
                    for(size_t i=0;i<values.size();++i)values[i]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(float(int((i*31+7)%127)-63)/511));
                    std::vector<uint8_t> packed(ggml_nbytes(t));
                    assert(ggml_quantize_chunk(GGML_TYPE_Q8_0,values.data(),packed.data(),0,t->ne[1],t->ne[0],nullptr)==packed.size());
                    ggml_backend_tensor_set(t,packed.data(),0,packed.size());
                } else {
                    std::vector<float> values(t->ne[0]);
                    const bool scale=entry.first.find("layer_norm.weight")!=std::string::npos;
                    for(size_t i=0;i<values.size();++i)values[i]=scale?1.f+float(int(i%7)-3)/32:float(int(i%17)-8)/257;
                    ggml_backend_tensor_set(t,values.data(),0,values.size()*4);
                }
            }
            auto * n=e.norm(prefix+".final_layer_norm",input);
            auto * raw=ggml_mul_mat(e.graph_ctx,w1,n);
            auto * expanded=ggml_map_custom2_inplace(e.graph_ctx,raw,e.w(prefix+".fc1.bias"),encoder_bias_gelu,GGML_N_TASKS_MAX,nullptr);
            auto * expected=e.linear_residual(prefix+".fc2",expanded,input);ggml_set_output(expected);
            auto * actual=e.ffn_tiled(prefix,input);ggml_set_output(actual);
            auto * ga=ggml_new_graph_custom(e.graph_ctx,max_nodes,false),*gb=ggml_new_graph_custom(e.graph_ctx,max_nodes,false);
            ggml_build_forward_expand(ga,expected);ggml_build_forward_expand(gb,actual);
            assert(ggml_graph_n_nodes(ga)<=int(max_nodes)&&ggml_graph_n_nodes(gb)<=int(max_nodes));
            int joins=0;for(int i=0;i<ggml_graph_n_nodes(gb);++i)joins+=ggml_graph_node(gb,i)->op==GGML_OP_SET;
            assert(joins==(rows+127)/128&&actual->view_src&&actual->src[0]&&actual->src[1]);
            auto * aa=ggml_gallocr_new(ggml_backend_get_default_buffer_type(e.backend));
            auto * ab=ggml_gallocr_new(ggml_backend_get_default_buffer_type(e.backend));
            assert(ggml_gallocr_alloc_graph(aa,ga)&&ggml_gallocr_alloc_graph(ab,gb));
            max_arena=std::max(max_arena,ggml_gallocr_get_buffer_size(ab,0));
            assert(actual->data!=input->data&&expected->data!=input->data&&actual->data!=expected->data);
            auto params=ggml_threadpool_params_default(threads);params.poll=0;e.pool=ggml_threadpool_new(&params);assert(e.pool);
            auto pa=ggml_graph_plan(ga,threads,e.pool),pb=ggml_graph_plan(gb,threads,e.pool);
            const size_t scratch_size=std::max(pa.work_size,pb.work_size);std::vector<uint8_t> scratch(scratch_size+128,0xa5);
            pa.work_data=pb.work_data=scratch.data()+64;
            for(int repeat=0;repeat<3;++repeat) {
                std::vector<float> values(size_t(width)*rows);
                for(size_t i=0;i<values.size();++i)values[i]=float(int((i*13+repeat*71)%997)-498)/173;
                values[0]=repeat?-0.f:0.f;
                ggml_backend_tensor_set(input,values.data(),0,values.size()*4);
                std::vector<float> poison(values.size(),-12345.f);ggml_backend_tensor_set(actual,poison.data(),0,poison.size()*4);
                assert(ggml_graph_compute(ga,&pa)==GGML_STATUS_SUCCESS&&ggml_graph_compute(gb,&pb)==GGML_STATUS_SUCCESS);
                assert(std::memcmp(actual->data,expected->data,ggml_nbytes(actual))==0);
                assert(std::memcmp(input->data,values.data(),values.size()*4)==0);
                for(size_t i=0;i<64;++i)assert(scratch[i]==0xa5&&scratch[64+scratch_size+i]==0xa5);
                ++executions;
            }
            std::fprintf(stderr,"FFN_TILE_ARENA rows=%d old=%zu tiled=%zu oldnodes=%d tilednodes=%d\n",rows,
                ggml_gallocr_get_buffer_size(aa,0),ggml_gallocr_get_buffer_size(ab,0),ggml_graph_n_nodes(ga),ggml_graph_n_nodes(gb));
            ggml_gallocr_free(aa);ggml_gallocr_free(ab);
        }
    }
    FFNFixture fixture;cpu_encoder_config config{384,4,6,1536,80,1500,4};
    for(bool refuse:{false,true}) {
        refuse_tiled=refuse;Encoder owner(config,true);cpu_encoder_memory memory{};
        owner.prepare(fixture.descriptors.data(),fixture.descriptors.size(),&memory);
        assert(owner.tiled_ffn==!refuse);int joins=0;
        for(int i=0;i<ggml_graph_n_nodes(owner.graph);++i)joins+=ggml_graph_node(owner.graph,i)->op==GGML_OP_SET;
        assert(joins==(refuse?0:48));
        assert(ggml_graph_n_nodes(owner.graph)<int(max_nodes));
        std::fprintf(stderr,"FFN_TILE_PREPARE refused=%d joins=%d nodes=%d graph=%llu scratch=%llu\n",
            refuse,joins,ggml_graph_n_nodes(owner.graph),(unsigned long long)memory.graph,(unsigned long long)memory.scratch);
    }
    assert(refused>0);
    std::printf("{\"status\":\"PASS\",\"executions\":%d,\"max_tiled_ffn_arena_bytes\":%zu,\"changed_inputs_exact\":true,\"source_immutable\":true}\n",executions,max_arena);
}
