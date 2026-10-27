// Actual pinned CPU ADD kernels, in-place projection and immutable residual.
#include "../encoder.cpp"
#include <cassert>
#include <limits>

int main() {
    int executions=0;
    for(int width:{17,384}) for(int rows:{1,4,17,1500}) for(int threads:{1,2,4}) for(int stride:{1,2}) {
        auto * ctx=ggml_init({2u<<20,nullptr,true});assert(ctx);
        auto * backend=ggml_backend_cpu_init();assert(backend);
        auto * input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,rows);ggml_set_input(input);
        auto * bias=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,width);ggml_set_input(bias);
        auto * residual_root=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,rows*stride);ggml_set_input(residual_root);
        auto * residual=ggml_view_2d(ctx,residual_root,width,rows,width*sizeof(float)*stride,0);
        auto * expected=ggml_add(ctx,residual,ggml_add(ctx,input,bias));ggml_set_output(expected);
        auto * raw=ggml_dup(ctx,input);
        auto * actual=ggml_map_custom3_inplace(ctx,raw,bias,residual,encoder_bias_residual,GGML_N_TASKS_MAX,nullptr);ggml_set_output(actual);
        auto * graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,expected);ggml_build_forward_expand(graph,actual);
        assert(actual->src[0]==raw && actual->src[1]==bias && actual->src[2]==residual && actual->view_src==raw);
        auto * allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));assert(ggml_gallocr_alloc_graph(allocator,graph));
        assert(actual->data==raw->data && actual->data!=input->data && actual->data!=expected->data && actual->data!=bias->data && actual->data!=residual_root->data);
        auto params=ggml_threadpool_params_default(threads);params.poll=0;auto * pool=ggml_threadpool_new(&params);assert(pool);
        auto plan=ggml_graph_plan(graph,threads,pool);std::vector<uint8_t> scratch(plan.work_size+128,0xa5);plan.work_data=scratch.data()+64;
        for(int repeat=0;repeat<4;++repeat) {
            std::vector<float> values(size_t(width)*rows),b(width),r(size_t(width)*rows*stride);
            for(size_t i=0;i<values.size();++i)values[i]=float(int((i*13+repeat*17)%127)-63)/8;
            for(int i=0;i<width;++i)b[i]=float((i*7+repeat*11)%17-8)/32;
            for(size_t i=0;i<r.size();++i)r[i]=float(int((i*11+repeat*19)%97)-48)/16;
            if(repeat>=2) {values[0]=-0.f;b[0]=-0.f;r[0]=-0.f;values[1]=0.f;b[1]=-0.f;r[1]=0.f;}
            if(repeat==3) {
                values[2]=std::numeric_limits<float>::infinity();r[2]=-std::numeric_limits<float>::infinity();
                values[3]=-std::numeric_limits<float>::infinity();r[3]=std::numeric_limits<float>::infinity();
                uint32_t bits=0xffc12345;std::memcpy(&values[4],&bits,4);bits=0x7fc54321;std::memcpy(&r[4],&bits,4);
                bits=0xff812345;std::memcpy(&values[5],&bits,4);bits=0x7f854321;std::memcpy(&r[5],&bits,4);
                b[6]=std::numeric_limits<float>::quiet_NaN();values[7]=std::numeric_limits<float>::denorm_min();
                r[7]=-std::numeric_limits<float>::denorm_min();
                // Detect reassociation: residual + (projection + bias) must remain ordered.
                values[8]=1.e20f;b[8]=-1.e20f;r[8]=3.25f;
            }
            ggml_backend_tensor_set(input,values.data(),0,values.size()*sizeof(float));ggml_backend_tensor_set(bias,b.data(),0,b.size()*sizeof(float));
            ggml_backend_tensor_set(residual_root,r.data(),0,r.size()*sizeof(float));
            assert(ggml_graph_compute(graph,&plan)==GGML_STATUS_SUCCESS);
            assert(std::memcmp(actual->data,expected->data,ggml_nbytes(actual))==0);
            assert(std::memcmp(input->data,values.data(),ggml_nbytes(input))==0 && std::memcmp(bias->data,b.data(),ggml_nbytes(bias))==0);
            assert(std::memcmp(residual_root->data,r.data(),ggml_nbytes(residual_root))==0);
            for(size_t i=0;i<64;++i)assert(scratch[i]==0xa5 && scratch[64+plan.work_size+i]==0xa5);
            ++executions;
        }
        ggml_threadpool_free(pool);ggml_gallocr_free(allocator);ggml_free(ctx);ggml_backend_free(backend);
    }
    std::printf("{\"status\":\"ENCODER_BIAS_RESIDUAL_PASS\",\"executions\":%d,\"bitwise_equal\":true,\"residual_immutable\":true}\n",executions);
}
