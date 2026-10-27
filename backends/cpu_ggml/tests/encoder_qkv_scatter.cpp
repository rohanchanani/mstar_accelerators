// Actual CUSTOM/typed aliases/gallocr/FLASH. Not model qualification/timing.
#include "../encoder.cpp"
#include <cassert>
#include <limits>

int main() {
    for(int threads:{1,2,4}) for(float key_bias:{0.f,-0.f,.125f}) {
        cpu_encoder_config c{};c.width=384;c.heads=6;c.positions=5;
        Encoder e(c,true);e.backend=ggml_backend_cpu_init();assert(e.backend);
        e.graph_ctx=ggml_init({4u<<20,nullptr,true});assert(e.graph_ctx);e.graph=ggml_new_graph(e.graph_ctx);
        const int d=c.width,padded=GGML_PAD(c.positions,256);
        auto * input=ggml_new_tensor_2d(e.graph_ctx,GGML_TYPE_F32,3*d,c.positions);ggml_set_input(input);ggml_set_output(input);
        ggml_tensor * plain[3],*biases[3];
        float kb[384];for(auto & x:kb)x=key_bias;bool skip=positive_zero_bias(kb,sizeof(kb));
        for(int i=0;i<3;++i) {
            plain[i]=ggml_new_tensor_2d(e.graph_ctx,GGML_TYPE_F32,d,c.positions);ggml_set_input(plain[i]);
            biases[i]=ggml_new_tensor_1d(e.graph_ctx,GGML_TYPE_F32,d);ggml_set_input(biases[i]);
            e.weights.emplace(std::string("test.")+(i==0?"q_proj":i==1?"k_proj":"v_proj")+".bias",biases[i]);
        }
        e.padded_k=ggml_new_tensor_1d(e.graph_ctx,GGML_TYPE_F16,d*padded);
        e.padded_v=ggml_new_tensor_1d(e.graph_ctx,GGML_TYPE_F16,d*padded);
        for(auto * t:{e.padded_k,e.padded_v}){ggml_set_input(t);ggml_set_output(t);}
        auto * actual=e.fused_attention(input,"test",skip);ggml_set_output(actual);ggml_build_forward_expand(e.graph,actual);
        ggml_tensor * expected_inputs[3];
        for(int i=0;i<3;++i)expected_inputs[i]=(i==1 && skip)?plain[i]:ggml_add(e.graph_ctx,plain[i],biases[i]);
        auto * expected=e.attention(expected_inputs[0],expected_inputs[1],expected_inputs[2]);
        ggml_set_output(expected);ggml_build_forward_expand(e.graph,expected);
        ggml_tensor * storage=nullptr,*flash=nullptr;
        for(int i=0;i<ggml_graph_n_nodes(e.graph);++i) {
            auto * node=ggml_graph_node(e.graph,i);
            if(node->op==GGML_OP_CUSTOM){assert(!storage);storage=node;}
            if(node->op==GGML_OP_FLASH_ATTN_EXT && !flash)flash=node;
        }
        assert(storage && flash && storage->type==GGML_TYPE_I8);
        // Retain diagnostic plane bytes beyond their last consumer so the test
        // may inspect them; production uses normal graph liveness.
        ggml_set_output(storage);
        const size_t qbytes=size_t(d)*c.positions*sizeof(float),kvbytes=size_t(d)*padded*sizeof(ggml_fp16_t);
        assert(ggml_nbytes(storage)==qbytes+2*kvbytes);
        for(int i=0;i<3;++i) {
            auto * view=flash->src[i];
            assert(view->view_src==storage && view->view_offs==(i==0?0:i==1?qbytes:qbytes+kvbytes));
            assert(view->ne[0]==64 && view->ne[1]==(i==0?c.positions:padded) && view->ne[2]==6);
            assert(view->nb[0]==size_t(i==0?4:2) && view->nb[1]==size_t(d)*(i==0?4:2) && view->nb[2]==size_t(64)*(i==0?4:2));
        }
        e.allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(e.backend));assert(ggml_gallocr_alloc_graph(e.allocator,e.graph));
        for(int i=0;i<3;++i) {
            auto * view=flash->src[i];assert(view->data==static_cast<uint8_t*>(storage->data)+view->view_offs);
            assert(uintptr_t(view->data)%(i==0?alignof(float):alignof(ggml_fp16_t))==0);
        }
        std::vector<ggml_fp16_t> zeros(d*padded,0);
        for(auto * t:{e.padded_k,e.padded_v})ggml_backend_tensor_set(t,zeros.data(),0,ggml_nbytes(t));
        for(int repeat=0;repeat<4;++repeat) {
            std::vector<float> joined(size_t(3*d)*c.positions),values[3],bias[3];
            for(int i=0;i<3;++i){values[i].resize(d*c.positions);bias[i].resize(d);}
            for(int t=0;t<c.positions;++t)for(int i=0;i<3;++i)for(int x=0;x<d;++x)
                joined[t*3*d+i*d+x]=values[i][t*d+x]=float((t*7+i*11+x*3+repeat*13)%31-15)/16;
            values[1][0]=joined[d]=-0.f; // +0 bias must not erase negative zero.
            for(int i=0;i<3;++i)for(int x=0;x<d;++x)bias[i][x]=i==1?key_bias:float((x%7)-3)/32;
            if(repeat==3) {values[2][1]=joined[2*d+1]=std::numeric_limits<float>::infinity();values[2][2]=joined[2*d+2]=std::numeric_limits<float>::quiet_NaN();}
            ggml_backend_tensor_set(input,joined.data(),0,joined.size()*sizeof(float));
            for(int i=0;i<3;++i) {
                ggml_backend_tensor_set(plain[i],values[i].data(),0,values[i].size()*sizeof(float));
                // Bias K is also an explicit CUSTOM src, even on skip branch.
                ggml_backend_tensor_set(biases[i],bias[i].data(),0,bias[i].size()*sizeof(float));
            }
            std::vector<uint8_t> poison(ggml_nbytes(storage),0xa5);ggml_backend_tensor_set(storage,poison.data(),0,poison.size());
            assert(ggml_graph_compute_with_ctx(e.graph_ctx,e.graph,threads)==GGML_STATUS_SUCCESS);
            std::vector<float> a(d*c.positions+2),b(d*c.positions+2);a.front()=b.front()=123;a.back()=b.back()=456;
            ggml_backend_tensor_get(actual,a.data()+1,0,d*c.positions*sizeof(float));ggml_backend_tensor_get(expected,b.data()+1,0,d*c.positions*sizeof(float));
            assert(a.front()==123 && a.back()==456 && b.front()==123 && b.back()==456);
            if(repeat<3)assert(std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);
            else for(size_t i=1;i+1<a.size();++i)assert(std::isnan(b[i])?std::isnan(a[i]):a[i]==b[i]);
            for(int i=0;i<3;++i) {
                auto * plane=static_cast<uint8_t*>(storage->data)+(i==0?0:i==1?qbytes:qbytes+kvbytes);
                for(int t=0;t<c.positions;++t)for(int x=0;x<d;++x) {
                    float value=values[i][t*d+x];if(i!=1 || !skip)value=value+bias[i][x];
                    if(i==0)assert(std::memcmp(plane+(t*d+x)*sizeof(float),&value,sizeof(float))==0);
                    else {ggml_fp16_t half=ggml_fp32_to_fp16(value);assert(std::memcmp(plane+(t*d+x)*sizeof(half),&half,sizeof(half))==0);}
                }
                if(i)for(size_t j=size_t(d)*c.positions*sizeof(ggml_fp16_t);j<kvbytes;++j)assert(plane[j]==0);
            }
            std::vector<float> unchanged(joined.size());ggml_backend_tensor_get(input,unchanged.data(),0,joined.size()*sizeof(float));assert(std::memcmp(unchanged.data(),joined.data(),joined.size()*sizeof(float))==0);
        }
        // Typed views reject invalid offset/type/extent before graph allocation.
        for(int kind=0;kind<3;++kind) {
            bool rejected=false;try {typed_plane(e.graph_ctx,storage,kind==0?GGML_TYPE_I8:GGML_TYPE_F32,d,c.positions,kind==1?1:ggml_nbytes(storage));}
            catch(const std::runtime_error &){rejected=true;}assert(rejected);
        }
    }
    std::puts("ENCODER_QKV_SCATTER_CONTROLS_PASS");
}
