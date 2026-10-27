// Upstream PAD -> F16 mechanism used by the public-padding encoder candidate.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cassert>
#include <cmath>
#include <limits>
int main() {
    auto * ctx=ggml_init({1u<<20,nullptr,false}); assert(ctx);
    auto * source=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,8,3);
    auto * output=ggml_cast(ctx,ggml_pad(ctx,source,0,5,0,0),GGML_TYPE_F16);
    assert(output->ne[0]==8 && output->ne[1]==8 && ggml_is_contiguous(output));
    auto * graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,output);
    for (int repeat=0;repeat<2;++repeat) {
        auto * in=static_cast<float *>(source->data);
        for(int i=0;i<24;++i)in[i]=float(i+repeat*32)/8.f;
        in[0]=std::numeric_limits<float>::infinity();in[1]=-in[0];in[2]=std::numeric_limits<float>::quiet_NaN();
        assert(ggml_graph_compute_with_ctx(ctx,graph,2)==GGML_STATUS_SUCCESS);
        auto * actual=static_cast<ggml_fp16_t *>(output->data);
        for(int i=0;i<24;++i){auto value=ggml_fp16_to_fp32(actual[i]);assert(std::isnan(in[i])?std::isnan(value):value==in[i]);}
        for(int i=24;i<64;++i)assert(actual[i]==ggml_fp32_to_fp16(0.f));
    }
    ggml_free(ctx);
}
