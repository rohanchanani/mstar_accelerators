// Real pinned GGML mechanism control; not model qualification.
#include "../encoder.cpp"
#include <cassert>
#include <limits>
int main() {
    float zero[4]={0,0,0,0}; assert(positive_zero_bias(zero,sizeof(zero)));
    for(float value:{1.f,-1.f,-0.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        zero[2]=value; assert(!positive_zero_bias(zero,sizeof(zero)));
    }
    cpu_encoder_config config{};config.width=8;config.heads=2;config.positions=3;
    Encoder owner(config,true);
    owner.backend=ggml_backend_cpu_init();assert(owner.backend);
    owner.graph_ctx=ggml_init({4u<<20,nullptr,true});assert(owner.graph_ctx);
    owner.graph=ggml_new_graph(owner.graph_ctx);
    const int padded=GGML_PAD(config.positions,256);
    owner.padded_k=ggml_new_tensor_1d(owner.graph_ctx,GGML_TYPE_F16,config.width*padded);
    owner.padded_v=ggml_new_tensor_1d(owner.graph_ctx,GGML_TYPE_F16,config.width*padded);
    for(auto * t:{owner.padded_k,owner.padded_v}) {ggml_set_input(t);ggml_set_output(t);}
    ggml_tensor * inputs[2][3];ggml_tensor * actual[2];ggml_tensor * expected[2];
    for(int layer=0;layer<2;++layer) {
        for(auto & t:inputs[layer]) {t=ggml_new_tensor_2d(owner.graph_ctx,GGML_TYPE_F32,8,3);ggml_set_input(t);}
        actual[layer]=owner.attention(inputs[layer][0],inputs[layer][1],inputs[layer][2]);
        ggml_set_output(actual[layer]);ggml_build_forward_expand(owner.graph,actual[layer]);
        auto split=[&](ggml_tensor * t,int n){return ggml_permute(owner.graph_ctx,ggml_reshape_3d(owner.graph_ctx,t,4,2,n),0,2,1,3);};
        auto * k=ggml_cast(owner.graph_ctx,ggml_pad(owner.graph_ctx,inputs[layer][1],0,padded-3,0,0),GGML_TYPE_F16);
        auto * v=ggml_cast(owner.graph_ctx,ggml_pad(owner.graph_ctx,inputs[layer][2],0,padded-3,0,0),GGML_TYPE_F16);
        expected[layer]=ggml_flash_attn_ext(owner.graph_ctx,split(inputs[layer][0],3),split(k,padded),split(v,padded),nullptr,.5f,0,0);
        ggml_set_output(expected[layer]);ggml_build_forward_expand(owner.graph,expected[layer]);
    }
    owner.allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(owner.backend));assert(ggml_gallocr_alloc_graph(owner.allocator,owner.graph));
    std::vector<ggml_fp16_t> zeros(8*padded,ggml_fp32_to_fp16(0));
    for(auto * t:{owner.padded_k,owner.padded_v})ggml_backend_tensor_set(t,zeros.data(),0,ggml_nbytes(t));
    for(int repeat=0;repeat<4;++repeat) {
        for(int layer=0;layer<2;++layer)for(int slot=0;slot<3;++slot) {
            float values[24];for(int i=0;i<24;++i)values[i]=float((i*7+layer*3+repeat*11+slot*5)%29-14)/16;
            if(repeat==3 && slot==2) {values[0]=std::numeric_limits<float>::infinity();values[1]=std::numeric_limits<float>::quiet_NaN();}
            ggml_backend_tensor_set(inputs[layer][slot],values,0,sizeof(values));
        }
        assert(ggml_graph_compute_with_ctx(owner.graph_ctx,owner.graph,2)==GGML_STATUS_SUCCESS);
        for(int layer=0;layer<2;++layer) {
            float a[24],b[24];ggml_backend_tensor_get(actual[layer],a,0,sizeof(a));ggml_backend_tensor_get(expected[layer],b,0,sizeof(b));
            if(repeat<3)assert(std::memcmp(a,b,sizeof(a))==0);
            else for(int i=0;i<24;++i)assert(std::isnan(b[i])?std::isnan(a[i]):a[i]==b[i]);
        }
        for(auto * t:{owner.padded_k,owner.padded_v}) {
            std::vector<ggml_fp16_t> data(8*padded);ggml_backend_tensor_get(t,data.data(),0,ggml_nbytes(t));
            for(int i=24;i<8*padded;++i)assert(data[i]==0);
        }
    }
}
