#pragma once
// Physical Q1/D64/F16 chunk specialization of pinned GGML d1be6f FLASH.
// Same split-KV planner, partial reducer, interleaved borrowed KV and online
// F16 accumulator. Normal-domain exp uses the existing attributed XNN core.
#include "flash_cross_exp_horner.h"
#if defined(MSTAR_CROSS_Q1_DIAGNOSTIC)
#include <atomic>
static std::atomic<uint64_t> mstar_cross_fast{0}, mstar_cross_fallback{0};
extern "C" uint64_t mstar_cross_q1_fast_count() { return mstar_cross_fast.load(); }
extern "C" uint64_t mstar_cross_q1_fallback_count() { return mstar_cross_fallback.load(); }
#endif

static __attribute__((noinline)) bool mstar_cross_q1_chunk(
        const ggml_compute_params * params, ggml_tensor * dst,
        int ir0, int ir1, int64_t begin, int64_t end,
        float * partials, int64_t partial_stride) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(__ARM_FEATURE_SVE)
    GGML_UNUSED(params);
    const auto * q = dst->src[0]; const auto * k = dst->src[1];
    const auto * v = dst->src[2]; const auto * mask = dst->src[3];
    float scale, bias, cap;
    memcpy(&scale, dst->op_params, 4); memcpy(&bias, dst->op_params+1, 4);
    memcpy(&cap, dst->op_params+2, 4);
    uint64_t fpcr; asm volatile("mrs %0, fpcr" : "=r"(fpcr));
    if (fpcr != 0 || !partials || dst->src[4] || bias != 0 || cap != 0 ||
        !std::isfinite(scale) || q->type != GGML_TYPE_F32 ||
        k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 ||
        q->ne[0] != 64 || q->ne[1] != 1 || q->ne[3] != 1 ||
        k->ne[0] != 64 || v->ne[0] != 64 || k->ne[1] != 1500 ||
        v->ne[1] != 1500 || k->nb[0] != 2 || v->nb[0] != 2 ||
        q->nb[0] != 4 || k->ne[3] != 1 || v->ne[3] != 1 ||
        k->ne[2] <= 0 || v->ne[2] <= 0 || q->ne[2]%k->ne[2] || q->ne[2]%v->ne[2] ||
        ir0 < 0 || ir1 < ir0 || ir1 > q->ne[2] || partial_stride < 66 ||
        begin < 0 || end > 1500 || begin > end)
        return false;
    const int64_t rk = q->ne[2]/k->ne[2], rv = q->ne[2]/v->ne[2];
    for (int h = ir0; h < ir1; ++h) {
        ggml_fp16_t packed_q[64];
        ggml_fp32_to_fp16_row((const float *)((const char *)q->data+h*q->nb[2]), packed_q, 64);
        const float16x8_t q0=vld1q_f16((const __fp16 *)packed_q+0), q1=vld1q_f16((const __fp16 *)packed_q+8);
        const float16x8_t q2=vld1q_f16((const __fp16 *)packed_q+16), q3=vld1q_f16((const __fp16 *)packed_q+24);
        const float16x8_t q4=vld1q_f16((const __fp16 *)packed_q+32), q5=vld1q_f16((const __fp16 *)packed_q+40);
        const float16x8_t q6=vld1q_f16((const __fp16 *)packed_q+48), q7=vld1q_f16((const __fp16 *)packed_q+56);
        float16x8_t a0=vdupq_n_f16(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
        const auto * mp = mask ? (const ggml_fp16_t *)((const char *)mask->data+(h%mask->ne[2])*mask->nb[2]) : nullptr;
        float M=-INFINITY, S=0;
        const char * kp=(const char *)k->data+(h/rk)*k->nb[2];
        const char * vp=(const char *)v->data+(h/rv)*v->nb[2];
        for (int64_t i=begin; i<end; ++i) {
            const float mv=mp ? GGML_CPU_FP16_TO_FP32(mp[i]) : 0;
            if (mv == -INFINITY) continue;
            const __fp16 * x=(const __fp16 *)(kp+i*k->nb[1]);
            // Exactly the pinned ARM F16 dot's four accumulators/two steps.
            float16x8_t d0=vfmaq_f16(vdupq_n_f16(0),vld1q_f16(x),q0), d1=vfmaq_f16(vdupq_n_f16(0),vld1q_f16(x+8),q1);
            float16x8_t d2=vfmaq_f16(vdupq_n_f16(0),vld1q_f16(x+16),q2), d3=vfmaq_f16(vdupq_n_f16(0),vld1q_f16(x+24),q3);
            d0=vfmaq_f16(d0,vld1q_f16(x+32),q4); d1=vfmaq_f16(d1,vld1q_f16(x+40),q5);
            d2=vfmaq_f16(d2,vld1q_f16(x+48),q6); d3=vfmaq_f16(d3,vld1q_f16(x+56),q7);
            d0=vaddq_f16(vaddq_f16(d0,d2),vaddq_f16(d1,d3));
            float s=vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(d0)),vcvt_f32_f16(vget_high_f16(d0))));
            s=s*scale; s+=mv;
            if (!std::isfinite(s)) return false;
            float ms=1, vs=1;
            if (s > M) {
                const float delta=M-s;
                if (M == -INFINITY) ms=0;
                else { if (delta < -87) return false; ms=vgetq_lane_f32(mstar_cross_exp_horner(vdupq_n_f32(delta)),0); }
                M=s;
                const float16x8_t z=vdupq_n_f16(ms);
                a0=vmulq_f16(a0,z); a1=vmulq_f16(a1,z); a2=vmulq_f16(a2,z); a3=vmulq_f16(a3,z);
                a4=vmulq_f16(a4,z); a5=vmulq_f16(a5,z); a6=vmulq_f16(a6,z); a7=vmulq_f16(a7,z);
            } else {
                const float delta=s-M;
                if (delta < -87) return false;
                vs=vgetq_lane_f32(mstar_cross_exp_horner(vdupq_n_f32(delta)),0);
            }
            const __fp16 * y=(const __fp16 *)(vp+i*v->nb[1]);
            const float16x8_t z=vdupq_n_f16(vs);
            a0=vfmaq_f16(a0,vld1q_f16(y),z); a1=vfmaq_f16(a1,vld1q_f16(y+8),z);
            a2=vfmaq_f16(a2,vld1q_f16(y+16),z); a3=vfmaq_f16(a3,vld1q_f16(y+24),z);
            a4=vfmaq_f16(a4,vld1q_f16(y+32),z); a5=vfmaq_f16(a5,vld1q_f16(y+40),z);
            a6=vfmaq_f16(a6,vld1q_f16(y+48),z); a7=vfmaq_f16(a7,vld1q_f16(y+56),z);
            S=S*ms+vs;
        }
        float result[64];
#define MSTAR_STORE(j,a) vst1q_f32(result+j,vcvt_f32_f16(vget_low_f16(a))); vst1q_f32(result+j+4,vcvt_f32_f16(vget_high_f16(a)))
        MSTAR_STORE(0,a0);MSTAR_STORE(8,a1);MSTAR_STORE(16,a2);MSTAR_STORE(24,a3);
        MSTAR_STORE(32,a4);MSTAR_STORE(40,a5);MSTAR_STORE(48,a6);MSTAR_STORE(56,a7);
#undef MSTAR_STORE
        if (!std::isfinite(S)) return false;
        for (float x : result) if (!std::isfinite(x)) return false;
        float * out=partials+h*partial_stride;
        out[0]=M; out[1]=S; memcpy(out+2,result,sizeof(result));
    }
    return true;
#else
    GGML_UNUSED(params);GGML_UNUSED(dst);GGML_UNUSED(ir0);GGML_UNUSED(ir1);
    GGML_UNUSED(begin);GGML_UNUSED(end);GGML_UNUSED(partials);GGML_UNUSED(partial_stride);
    return false;
#endif
}
