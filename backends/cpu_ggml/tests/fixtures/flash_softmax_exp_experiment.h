#pragma once
// Physical operator experiment. Original GGML handles unsupported rows;
// guarded normal rows choose exact GGML or pinned XNNPACK arithmetic.
#include "vec.h"
// Bounded physical experiment. Variant1 retains exact GGML normal arithmetic;
// variant2 uses pinned XNNPACK rr1-p5 coefficients and operation order.
// The original GGML path handles every row outside the proven normal range.
#include <cmath>
#include <cstring>
#if MSTAR_SOFTMAX_EXP_VARIANT != 1 && MSTAR_SOFTMAX_EXP_VARIANT != 2
#error "Select guardedGGML1 or guardedXNN2"
#endif
#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(__ARM_FEATURE_SVE)
static inline float32x4_t mstar_exp_normal(float32x4_t x) {
#if MSTAR_SOFTMAX_EXP_VARIANT == 1
    // Exact normal return path of pinned ggml_v_expf; same intrinsic ordering.
    const float32x4_t r=vdupq_n_f32(0x1.8p23f);
    const float32x4_t z=vfmaq_f32(r,x,vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n=vsubq_f32(z,r);
    const float32x4_t b=vfmsq_f32(vfmsq_f32(x,n,vdupq_n_f32(0x1.62e4p-1f)),n,vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e=vshlq_n_u32(vreinterpretq_u32_f32(z),23);
    const float32x4_t k=vreinterpretq_f32_u32(vaddq_u32(e,vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const float32x4_t u=vmulq_f32(b,b);
    const float32x4_t j=vfmaq_f32(vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f),b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f),vdupq_n_f32(0x1.555e66p-3f),b),
        vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f),vdupq_n_f32(0x1.0e4020p-7f),b),u),u);
    return vfmaq_f32(k,j,k);
#else
    // Copyright 2020 Google LLC. BSD-3-Clause; see XNNPACK_EXP_LICENSE.
    // google/XNNPACK941f80cb44b206ec9dc32622ae499c9c81f0f980
    // src/f32-raddstoreexpminusmax/neon-p5.c.in FMA=true normal-domain core.
    const float32x4_t bias=vdupq_n_f32(0x1.8000FEp23f);
    float32x4_t n=vfmaq_f32(bias,x,vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t scale=vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(n),23));
    n=vsubq_f32(n,bias);
    float32x4_t t=vfmaq_f32(x,n,vdupq_n_f32(-0x1.62E430p-1f));
    float32x4_t p=vfmaq_f32(vdupq_n_f32(0x1.573A1Ap-5f),vdupq_n_f32(0x1.0F9F9Cp-7f),t);
    p=vfmaq_f32(vdupq_n_f32(0x1.555A80p-3f),p,t);
    p=vfmaq_f32(vdupq_n_f32(0x1.FFFDC6p-2f),p,t);
    p=vfmaq_f32(vdupq_n_f32(0x1.FFFFF6p-1f),p,t);
    t=vmulq_f32(t,scale);
    return vfmaq_f32(scale,p,t);
#endif
}
#endif


static inline bool mstar_exp_finite_max64(const float * x, float * result, float * absmax) {
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
    memcpy(absmax,&magnitude_bits,4);
    *result=value;return true;
#else
    GGML_UNUSED(x);GGML_UNUSED(result);GGML_UNUSED(absmax);return false;
#endif
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static void mstar_flash_softmax_exp64(float * GGML_RESTRICT scores,
                                  float * GGML_RESTRICT values,
                                  float * GGML_RESTRICT sums,
                                  float * GGML_RESTRICT maxima,
                                  bool * GGML_RESTRICT skip) {
    // All five spans are distinct fixed parts of the existing tile workspace
    // or stack. No native/resource owner or additional allocation is created.
    for (int row=0;row<64;++row) {
        float * x=scores+row*64;
        float tile_max, absolute_max=0;
        const bool vector_row=mstar_exp_finite_max64(x,&tile_max,&absolute_max);
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
            const bool normal_range=std::isfinite(next) && -absolute_max-next>=-87.f;
            if(normal_range) {
                for(int i=0;i<64;i+=8) {
                    const float32x4_t x0=mstar_exp_normal(vsubq_f32(vld1q_f32(x+i),vdupq_n_f32(next)));
                    const float32x4_t x1=mstar_exp_normal(vsubq_f32(vld1q_f32(x+i+4),vdupq_n_f32(next)));
                    vst1q_f32(x+i,x0);vst1q_f32(x+i+4,x1);
                    a=vaddq_f32(a,x0);b=vaddq_f32(b,x1);
                }
            } else {
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
