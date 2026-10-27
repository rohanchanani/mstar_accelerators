// Complete old/new CUSTOM callbacks under the actual GGML team. Includes graph
// dispatch/barriers. Model matmuls and external fixture upload are not timed.
#include "../encoder.cpp"
#include "ggml-cpu/vec.h"
#include <cassert>
#include <chrono>

int main() {
    assert(encoder_gelu_lut_native());
    constexpr int width=1536;
    for (int rows : {128,92,1500}) for (int block=0;block<4;++block) {
        const bool candidate=block==1 || block==2;
        auto * ctx=ggml_init({2u<<20,nullptr,true}); assert(ctx);
        auto * backend=ggml_backend_cpu_init(); assert(backend);
        auto * input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,rows); ggml_set_input(input);
        auto * bias=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,width); ggml_set_input(bias);
        auto * output=ggml_map_custom2_inplace(ctx,input,bias,
            candidate ? encoder_bias_gelu : encoder_bias_gelu_compiled,
            GGML_N_TASKS_MAX,nullptr); ggml_set_output(output);
        auto * graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,output);
        auto * allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        assert(ggml_gallocr_alloc_graph(allocator,graph));
        assert(output->data==input->data && output->view_src==input);
        auto params=ggml_threadpool_params_default(4);params.poll=0;
        auto * pool=ggml_threadpool_new(&params); assert(pool);
        auto plan=ggml_graph_plan(graph,4,pool);
        std::vector<uint8_t> scratch(plan.work_size+128,0xa5);plan.work_data=scratch.data()+64;
        std::vector<float> values(size_t(width)*rows),b(width),expected(values.size());
        for(int repeat=0;repeat<6;++repeat) {
            for(size_t i=0;i<values.size();++i) values[i]=float(int((i*13+repeat*17)%1021)-510)/32;
            for(int i=0;i<width;++i)b[i]=float((i*7+repeat*11)%17-8)/32;
            ggml_backend_tensor_set(input,values.data(),0,values.size()*4);
            ggml_backend_tensor_set(bias,b.data(),0,b.size()*4);
            const auto start=std::chrono::steady_clock::now();
            assert(ggml_graph_compute(graph,&plan)==GGML_STATUS_SUCCESS);
            const auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
            for(size_t i=0;i<values.size();++i) expected[i]=values[i]+b[i%width];
            ggml_vec_gelu_f32(int(expected.size()),expected.data(),expected.data());
            assert(std::memcmp(expected.data(),output->data,expected.size()*4)==0);
            assert(std::memcmp(b.data(),bias->data,b.size()*4)==0);
            for(size_t i=0;i<64;++i) assert(scratch[i]==0xa5 && scratch[64+plan.work_size+i]==0xa5);
            std::printf("{\"arm\":\"%s\",\"block\":%d,\"rows\":%d,\"width\":1536,\"threads\":4,\"repeat\":%d,\"wall_ns\":%lld,\"exact\":true}\n",
                        candidate?"candidate":"baseline",block,rows,repeat,(long long)elapsed);
        }
        ggml_threadpool_free(pool);ggml_gallocr_free(allocator);ggml_free(ctx);ggml_backend_free(backend);
    }
    std::printf("{\"status\":\"COMPLETE_CALLBACK_ABBA_PASS\",\"runs\":72,\"threads\":4,\"native\":true}\n");
}
