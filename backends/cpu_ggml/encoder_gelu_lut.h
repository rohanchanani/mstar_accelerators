#pragma once

#include "ggml.h"
#include <cstdint>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// The same immutable table initialized by pinned GGML CPU. Its mapping is
// ggml-cpu/vec.h::ggml_vec_gelu_f32 (GGML_GELU_FP16); no new approximation.
extern "C" { extern ggml_fp16_t ggml_table_gelu_f16[1 << 16]; }

inline bool encoder_gelu_lut_native() {
#if defined(__aarch64__) && defined(__ARM_NEON)
    uint64_t fpcr;
    asm volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr == 0;
#else
    return false;
#endif
}

#if defined(__aarch64__) && defined(__ARM_NEON)
// Caller selects only default FPCR. Half conversion is round-to-nearest-even,
// exactly the pinned software index conversion. Canonicalize NaN indices as
// the software primitive does, while retaining the biased input for clamps.
inline float32x4_t encoder_gelu_lut4(float32x4_t x) {
    const auto bits = vreinterpretq_u32_f32(x);
    const auto nan = vcgtq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffffff)),
                              vdupq_n_u32(0x7f800000));
    const auto low = vcleq_f32(x, vdupq_n_f32(-10.f));
    const auto high = vcgeq_f32(x, vdupq_n_f32(10.f));
    const auto skip = vorrq_u32(nan, vorrq_u32(low, high));
    const auto finite = vbslq_f32(skip, vdupq_n_f32(0.f), x);
    auto index = vreinterpret_u16_f16(vcvt_f16_f32(finite));
    // Only the sign is retained from a NaN, never its original payload.
    const auto canonical = vorr_u16(vand_u16(vshrn_n_u32(bits, 16), vdup_n_u16(0x8000)),
                                    vdup_n_u16(0x7e00));
    index = vbsl_u16(vmovn_u32(nan), canonical, index);
    uint16x4_t values = vdup_n_u16(0);
    values = vset_lane_u16(ggml_table_gelu_f16[vget_lane_u16(index, 0)], values, 0);
    values = vset_lane_u16(ggml_table_gelu_f16[vget_lane_u16(index, 1)], values, 1);
    values = vset_lane_u16(ggml_table_gelu_f16[vget_lane_u16(index, 2)], values, 2);
    values = vset_lane_u16(ggml_table_gelu_f16[vget_lane_u16(index, 3)], values, 3);
    auto result = vcvt_f32_f16(vreinterpret_f16_u16(values));
    result = vbslq_f32(low, vdupq_n_f32(0.f), result);
    return vbslq_f32(high, x, result);
}

inline void encoder_bias_gelu_lut_row(float * dst, const float * input,
                                     const float * bias, int width) {
    // Actual accelerated encoder widths are divisible by four. The wrapper
    // retains the original compiled implementation for every other geometry.
    for (int i = 0; i < width; i += 4) {
        const auto x = vaddq_f32(vld1q_f32(input+i), vld1q_f32(bias+i));
        vst1q_f32(dst+i, encoder_gelu_lut4(x));
    }
}
#endif
