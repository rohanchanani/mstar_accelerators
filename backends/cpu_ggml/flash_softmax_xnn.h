#pragma once
// Normal-domain exp core only. flash_softmax_tile.h owns row classification,
// the finite [-87,0] guard, online rescaling, accumulation and exceptional path.
// Pinned source/license: see FLASH_SOFTMAX_XNN.md and XNNPACK_EXP_LICENSE.
#include "vec.h"
#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(__ARM_FEATURE_SVE)
static inline float32x4_t mstar_exp_normal(float32x4_t x) {
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
}
#endif
