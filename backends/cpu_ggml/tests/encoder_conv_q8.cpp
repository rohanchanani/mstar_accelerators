// Actual GGML column orientation/zero padding/quantized-dot controls, not qualification.
#include "ggml.h"
#include "ggml-backend.h"
#include <cassert>
#include <cstring>
static bool refuse_conv=false;
static int refused=0;
static bool supported_for_test(ggml_backend_dev_t device,const ggml_tensor * op) {
    if(refuse_conv && op->op==GGML_OP_MUL_MAT && (op->src[0]->ne[0]==256||op->src[0]->ne[0]==1152)&&op->src[0]->ne[1]==384) {++refused;return false;}
    return ggml_backend_dev_supports_op(device,op);
}
#define ggml_backend_dev_supports_op supported_for_test
#include "../encoder.cpp"
#undef ggml_backend_dev_supports_op

struct Fixture {
    struct Value {std::vector<int64_t> shape;std::vector<float> data;};
    std::map<std::string,Value> values;
    std::vector<cpu_encoder_weight> descriptors;
    void add(const std::string & name,std::initializer_list<int64_t> shape,float value=0) {
        size_t n=1;for(auto x:shape)n*=x;values.emplace(name,Value{shape,std::vector<float>(n,value)});
    }
    Fixture() {
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
    int executions=0;
    auto * backend=ggml_backend_cpu_init();assert(backend);
    const auto * traits=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    assert(traits->vec_dot_type==GGML_TYPE_Q8_0 && traits->from_float);
    for(int channels:{80,384})for(int stride:{1,2})for(int time:{9,17,33})for(int threads:{1,2,4}) {
        const int k=3*channels,kp=GGML_PAD(k,32),out=16,rows=(time+stride-1)/stride;
        auto * inputs=ggml_init({2u<<20,nullptr,true});auto * ctx=ggml_init({2u<<20,nullptr,true});assert(inputs&&ctx);
        auto * image=ggml_new_tensor_2d(inputs,GGML_TYPE_F32,time,channels);ggml_set_input(image);
        auto * kernel=ggml_new_tensor_3d(inputs,GGML_TYPE_F16,3,channels,out);ggml_set_input(kernel);
        auto * weight=ggml_new_tensor_2d(inputs,GGML_TYPE_Q8_0,kp,out);ggml_set_input(weight);
        auto * buffer=ggml_backend_alloc_ctx_tensors(inputs,backend);assert(buffer);
        auto * result=convolution_q8(ctx,kernel,weight,image,stride);ggml_set_output(result);
        assert(result->ne[0]==out&&result->ne[1]==rows&&ggml_is_contiguous(result));
        auto * columns=result->src[1];ggml_set_output(columns);
        assert(columns->ne[0]==kp&&columns->ne[1]==rows&&ggml_is_contiguous(columns));
        auto * graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,result);
        auto * allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));assert(ggml_gallocr_alloc_graph(allocator,graph));
        std::vector<float> values(time*channels),weights(kp*out),decoded(kp*out),quant_columns(rows*kp);
        std::vector<uint8_t> canonical(ggml_nbytes(weight)),lhs(ggml_row_size(GGML_TYPE_Q8_0,kp));
        for(int repeat=0;repeat<2;++repeat) {
            for(size_t i=0;i<values.size();++i)values[i]=float(int((i*17+repeat*7)%101)-50)/64;
            for(int row=0;row<out;++row)for(int j=0;j<kp;++j)weights[row*kp+j]=j<k?ggml_fp16_to_fp32(ggml_fp32_to_fp16(float(int((row*13+j*7+repeat)%59)-29)/128)):0;
            assert(ggml_quantize_chunk(GGML_TYPE_Q8_0,weights.data(),canonical.data(),0,out,kp,nullptr)==canonical.size());
            for(int row=0;row<out;++row)ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(canonical.data()+row*ggml_row_size(GGML_TYPE_Q8_0,kp),decoded.data()+row*kp,kp);
            ggml_backend_tensor_set(image,values.data(),0,values.size()*4);ggml_backend_tensor_set(weight,canonical.data(),0,canonical.size());
            assert(ggml_graph_compute_with_ctx(ctx,graph,threads)==GGML_STATUS_SUCCESS);
            const auto * cols=static_cast<const float *>(columns->data),*actual=static_cast<const float *>(result->data);
            for(int row=0;row<rows;++row) {
                for(int j=0;j<kp;++j) {
                    const int chan=j/3,offset=j%3,position=row*stride+offset-1;
                    const float expected=j<k&&position>=0&&position<time?values[chan*time+position]:0;
                    assert(cols[row*kp+j]==expected);
                }
                traits->from_float(cols+row*kp,lhs.data(),kp);ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(lhs.data(),quant_columns.data()+row*kp,kp);
                for(int channel=0;channel<out;++channel) {
                    double expected=0;for(int j=0;j<kp;++j)expected+=double(decoded[channel*kp+j])*quant_columns[row*kp+j];
                    assert(std::abs(double(actual[row*out+channel])-expected)<=3e-5*(1+std::abs(expected)));
                }
            }
            assert(std::memcmp(image->data,values.data(),values.size()*4)==0 && std::memcmp(weight->data,canonical.data(),canonical.size())==0);
            ++executions;
        }
        ggml_gallocr_free(allocator);ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_free(inputs);
    }
    ggml_backend_free(backend);
    Fixture f;cpu_encoder_config config{384,4,6,1536,80,1500,2};
    for(bool refuse:{false,true}) {
        refuse_conv=refuse;Encoder e(config,true);cpu_encoder_memory memory{};e.prepare(f.descriptors.data(),f.descriptors.size(),&memory);
        int active=0,f16mm=0;
        for(int i=0;i<2;++i) {
            assert(bool(e.conv_q8[i])==bool(!refuse&&(MSTAR_ENCODER_Q8_CONV&(1<<i))));
            if(e.conv_q8[i]) {++active;assert(e.conv_q8[i]->ne[0]==(i?1152:256)&&e.conv_q8[i]->ne[1]==384&&e.conv_q8[i]->buffer);}
        }
        for(int i=0;i<ggml_graph_n_nodes(e.graph);++i) {
            auto * n=ggml_graph_node(e.graph,i);if(n->op==GGML_OP_MUL_MAT&&n->src[0]->type==GGML_TYPE_F16)++f16mm;
        }
        assert(f16mm==2-active && e.output->ne[0]==384 && e.output->ne[1]==1500);
        for(auto & v:f.values)for(float x:v.second.data)assert(x==(v.first.find("norm.weight")!=std::string::npos?1.f:0.f));
        std::printf("PREPARE selection=%d refuse=%d active=%d f16_matmul=%d weights=%llu graph=%llu scratch=%llu\n",MSTAR_ENCODER_Q8_CONV,refuse,active,f16mm,(unsigned long long)memory.weights,(unsigned long long)memory.graph,(unsigned long long)memory.scratch);
    }
    std::printf("ENCODER_CONV_Q8_CONTROLS_PASS executions=%d selection=%d refused=%d layout_padding_exact=1 quantized_fp64_reference=1 input_immutable=1\n",executions,MSTAR_ENCODER_Q8_CONV,refused);
}
