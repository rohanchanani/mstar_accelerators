#pragma once
// Physical encoder tile helper. Pinned GGML vec.h supplies both the original
// exceptional fallback and the same ARM exp polynomial/reduction operations.
#include "vec.h"
#if defined(MSTAR_ENCODER_SOFTMAX_XNN)
#include "flash_softmax_xnn.h"
#include <cmath>
#include <cstring>
#endif

static inline bool mstar_flash_finite_max64(const float * x, float * result
#if defined(MSTAR_ENCODER_SOFTMAX_XNN)
                                            , float * absmax
#endif
                                            ) {
#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(__ARM_FEATURE_SVE)
    float32x4_t maximum = vld1q_f32(x);
    uint32x4_t magnitude = vandq_u32(vreinterpretq_u32_f32(maximum), vdupq_n_u32(0x7fffffff));
    for (int i=4;i<64;i+=4) {
        const float32x4_t value=vld1q_f32(x+i);
        maximum=vmaxq_f32(maximum,value);
        magnitude=vmaxq_u32(magnitude,vandq_u32(vreinterpretq_u32_f32(value),vdupq_n_u32(0x7fffffff)));
    }
    const float value=vmaxvq_f32(maximum);
    // Original MAX is an ordered right-on-tie fold, including NaN and signed
    // zero behavior. Use it unchanged for every nonfinite row or zero max.
    const uint32_t magnitude_bits=vmaxvq_u32(magnitude);
    if (magnitude_bits>=UINT32_C(0x7f800000) || value==0.f) return false;
#if defined(MSTAR_ENCODER_SOFTMAX_XNN)
    memcpy(absmax,&magnitude_bits,4);
#endif
    *result=value;return true;
#else
    GGML_UNUSED(x);GGML_UNUSED(result);
#if defined(MSTAR_ENCODER_SOFTMAX_XNN)
    GGML_UNUSED(absmax);
#endif
    return false;
#endif
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static void mstar_flash_softmax64(float * GGML_RESTRICT scores,
                                  float * GGML_RESTRICT values,
                                  float * GGML_RESTRICT sums,
                                  float * GGML_RESTRICT maxima,
                                  bool * GGML_RESTRICT skip) {
    // All five spans are distinct fixed parts of the existing tile workspace
    // or stack. No native/resource owner or additional allocation is created.
    for (int row=0;row<64;++row) {
        float * x=scores+row*64;
        float tile_max;
#if defined(MSTAR_ENCODER_SOFTMAX_XNN)
        float absolute_max=0;
        const bool vector_row=mstar_flash_finite_max64(x,&tile_max,&absolute_max);
#else
        const bool vector_row=mstar_flash_finite_max64(x,&tile_max);
#endif
        if(!vector_row)ggml_vec_max_f32(64,&tile_max,x);
        if(tile_max==-INFINITY){skip[row]=true;continue;}
        const float previous=maxima[row], next=fmaxf(previous,tile_max);
        if(next>previous){
            const float factor=expf(previous-next);
            ggml_vec_scale_f32(64,values+row*64,factor);
            sums[row]*=factor;
        }
        maxima[row]=next;
#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(__ARM_FEATURE_SVE)
        if(vector_row){
            // Same exponentials; two independent vector sums remove sixteen
            // horizontal reductions and FP64 conversions from each finite row.
            // Finite rounding differs within the unchanged numerical policy.
            float32x4_t a=vdupq_n_f32(0),b=vdupq_n_f32(0);
#if defined(MSTAR_ENCODER_SOFTMAX_XNN)
            // This conservative row bound proves each rounded x-next is in
            // [-87,0]. Every other row retains the pinned GGML exp primitive.
            const bool normal_range=std::isfinite(next) && -absolute_max-next>=-87.f;
            if(normal_range) {
                for(int i=0;i<64;i+=8) {
                    const float32x4_t x0=mstar_exp_normal(vsubq_f32(vld1q_f32(x+i),vdupq_n_f32(next)));
                    const float32x4_t x1=mstar_exp_normal(vsubq_f32(vld1q_f32(x+i+4),vdupq_n_f32(next)));
                    vst1q_f32(x+i,x0);vst1q_f32(x+i+4,x1);
                    a=vaddq_f32(a,x0);b=vaddq_f32(b,x1);
                }
            } else
#endif
            {
            for(int i=0;i<64;i+=8){
                const float32x4_t x0=ggml_v_expf(vsubq_f32(vld1q_f32(x+i),vdupq_n_f32(next)));
                const float32x4_t x1=ggml_v_expf(vsubq_f32(vld1q_f32(x+i+4),vdupq_n_f32(next)));
                vst1q_f32(x+i,x0);vst1q_f32(x+i+4,x1);
                a=vaddq_f32(a,x0);b=vaddq_f32(b,x1);
            }
            }
            sums[row]+=(ggml_float)vaddvq_f32(vaddq_f32(a,b));
        }else
#endif
        {
            sums[row]+=ggml_vec_soft_max_f32(64,x,x,next);
        }
    }
}
