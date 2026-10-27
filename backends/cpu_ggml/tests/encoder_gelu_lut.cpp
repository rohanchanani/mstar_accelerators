// Actual pinned compiled CPU ADD/GELU and CUSTOM/arena comparison.
#include "../encoder.cpp"
#include <cassert>
#include <limits>

void graph_controls(bool old_callback_reference = false) {
    int executions=0;
    for(int rows:{1,4,17,92,128,1500}) for(int threads:{1,2,4}) {
        constexpr int width=1536;
        auto * ctx=ggml_init({2u<<20,nullptr,true});assert(ctx);
        auto * backend=ggml_backend_cpu_init();assert(backend);
        auto * input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,rows);ggml_set_input(input);
        auto * bias=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,width);ggml_set_input(bias);
        auto * expected=old_callback_reference ? ggml_map_custom2_inplace(ctx,ggml_dup(ctx,input),bias,encoder_bias_gelu_compiled,GGML_N_TASKS_MAX,nullptr) : ggml_gelu(ctx,ggml_add(ctx,input,bias));ggml_set_output(expected);
        auto * raw=ggml_dup(ctx,input);
        auto * actual=ggml_map_custom2_inplace(ctx,raw,bias,encoder_bias_gelu,GGML_N_TASKS_MAX,nullptr);ggml_set_output(actual);
        auto * graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,expected);ggml_build_forward_expand(graph,actual);
        assert(actual->src[0]==raw && actual->src[1]==bias && actual->view_src==raw && actual->type==GGML_TYPE_F32);
        auto * allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));assert(ggml_gallocr_alloc_graph(allocator,graph));
        assert(actual->data==raw->data && actual->data!=input->data && actual->data!=expected->data && actual->data!=bias->data);
        auto params=ggml_threadpool_params_default(threads);params.poll=0;auto * pool=ggml_threadpool_new(&params);assert(pool);
        auto plan=ggml_graph_plan(graph,threads,pool);std::vector<uint8_t> scratch(plan.work_size+128,0xa5);plan.work_data=scratch.data()+64;
        for(int repeat=0;repeat<4;++repeat) {
            std::vector<float> values(size_t(width)*rows),b(width);
            for(size_t i=0;i<values.size();++i)values[i]=float(int((i*13+repeat*17)%127)-63)/8;
            for(int i=0;i<width;++i)b[i]=float((i*7+repeat*11)%17-8)/32;
            if(repeat>=2) {values[0]=-0.f;b[0]=-0.f;values[1]=0.f;b[1]=0.f;}
            if(repeat==3) {
                values[2]=std::numeric_limits<float>::infinity();values[3]=-std::numeric_limits<float>::infinity();
                uint32_t bits=0xffc12345;std::memcpy(&values[4],&bits,4);bits=0x7f812345;std::memcpy(&values[5],&bits,4);
                b[6]=std::numeric_limits<float>::quiet_NaN();values[7]=std::numeric_limits<float>::denorm_min();
            }
            ggml_backend_tensor_set(input,values.data(),0,values.size()*sizeof(float));ggml_backend_tensor_set(bias,b.data(),0,b.size()*sizeof(float));
            assert(ggml_graph_compute(graph,&plan)==GGML_STATUS_SUCCESS);
            assert(std::memcmp(actual->data,expected->data,ggml_nbytes(actual))==0);
            assert(std::memcmp(input->data,values.data(),ggml_nbytes(input))==0 && std::memcmp(bias->data,b.data(),ggml_nbytes(bias))==0);
            for(size_t i=0;i<64;++i)assert(scratch[i]==0xa5 && scratch[64+plan.work_size+i]==0xa5);
            ++executions;
        }
        ggml_threadpool_free(pool);ggml_gallocr_free(allocator);ggml_free(ctx);ggml_backend_free(backend);
    }
    std::printf("{\"status\":\"ENCODER_BIAS_GELU_PASS\",\"executions\":%d,\"bitwise_equal\":true}\n",executions);
}


#include "ggml-cpu/vec.h"
#include <algorithm>
#include <chrono>

int main() {
    auto * backend = ggml_backend_cpu_init(); assert(backend);
    std::vector<float> values;
    auto bits = [&](uint32_t u) { float x; std::memcpy(&x,&u,4); values.push_back(x); };
    for(uint32_t h=0;h<65536;++h) values.push_back(ggml_fp16_to_fp32(ggml_fp16_t(h)));
    for(uint32_t h=0;h<0x7bff;++h) {
        const float middle = float((double(ggml_fp16_to_fp32(ggml_fp16_t(h))) + double(ggml_fp16_to_fp32(ggml_fp16_t(h+1))))*.5);
        for(float x:{std::nextafter(middle,-INFINITY),middle,std::nextafter(middle,INFINITY)}) { values.push_back(x);values.push_back(-x); }
    }
    for(float x:{-10.f,10.f,0.f,-0.f}) for(float y:{std::nextafter(x,-INFINITY),x,std::nextafter(x,INFINITY)}) values.push_back(y);
    for(uint32_t u:{1u,0x80000001u,0x007fffffu,0x807fffffu,0x00800000u,0x80800000u,0x7f800000u,0xff800000u,0x7f800001u,0xff800001u,0x7fc12345u,0xffc12345u,0xffffffffu}) bits(u);
    uint32_t seed=0x1434ae92;
    for(int i=0;i<4*1024*1024;++i) { seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;bits(seed); }
    while(values.size()%4) values.push_back(0);
    std::vector<float> expected(values.size()), actual(values.size());
    ggml_vec_gelu_f32(int(values.size()),expected.data(),values.data());
    bool native=encoder_gelu_lut_native();
#if defined(__aarch64__) && defined(__ARM_NEON)
    assert(native); // Actual acceleration, no environment setter or fake PASS.
    for(size_t i=0;i<values.size();i+=4) vst1q_f32(actual.data()+i,encoder_gelu_lut4(vld1q_f32(values.data()+i)));
    assert(std::memcmp(expected.data(),actual.data(),values.size()*4)==0);
#else
    assert(!native);
#endif
    std::printf("{\"status\":\"LUT_INDEX_CONTROL_PASS\",\"patterns\":%zu,\"native\":%s}\n",values.size(),native?"true":"false");
    ggml_backend_free(backend);
    graph_controls();
#if defined(__aarch64__) && defined(__ARM_NEON)
    uint64_t saved; asm volatile("mrs %0, fpcr" : "=r"(saved));
    // Test-only mode changes. Production only reads FPCR and never mutates it.
    for(uint64_t mode : {UINT64_C(1)<<22, UINT64_C(2)<<22, UINT64_C(3)<<22,
                         UINT64_C(1)<<24, UINT64_C(1)<<25, UINT64_C(1)<<19}) {
        const uint64_t selected = saved | mode;
        asm volatile("msr fpcr, %0" :: "r"(selected) : "memory");
        assert(!encoder_gelu_lut_native());
        graph_controls(true);
        asm volatile("msr fpcr, %0" :: "r"(saved) : "memory");
    }
    assert(encoder_gelu_lut_native());
#endif
    std::printf("{\"status\":\"GELU_LUT_ALL_CONTROLS_PASS\",\"patterns\":%zu,\"graph_executions\":%d,\"native\":%s}\n",values.size(),native?504:72,native?"true":"false");
}
