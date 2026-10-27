#pragma once
// Form the original two 32-key partials, then add to the existing output tile.
#include "repack.h"
static_assert(sizeof(block_q8_0x4)==136, "pinned four-row Q8 ABI changed");
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#endif

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((noinline,optimize("fp-contract=off")))
#elif defined(__clang__)
__attribute__((noinline))
#endif
static bool mstar_pv_exact_accumulate64(float * GGML_RESTRICT output,
                            const void * GGML_RESTRICT vx,const void * GGML_RESTRICT vy,
                            bool finite_history) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    // The caller establishes finite zero-origin history. Refused guards leave
    // output untouched so the original product-then-add path can run.
    if(!finite_history)return false;
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    uint64_t fpcr;asm volatile("mrs %0, fpcr":"=r"(fpcr));
    if(fpcr!=0)return false;
    {
        const auto * a=static_cast<const block_q8_0x4 *>(vy);
        const auto * b=static_cast<const block_q8_0x4 *>(vx);
        for(int y=0;y<16;++y)for(int x=0;x<16;x+=2) {
            float32x4_t f00=vdupq_n_f32(0),f01=vdupq_n_f32(0),f02=vdupq_n_f32(0),f03=vdupq_n_f32(0);
            float32x4_t f10=vdupq_n_f32(0),f11=vdupq_n_f32(0),f12=vdupq_n_f32(0),f13=vdupq_n_f32(0);
            // Keep the two 32-key groups in source order.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC unroll 1
#elif defined(__clang__)
#pragma clang loop unroll(disable)
#endif
            for(int l=0;l<2;++l) {
                const auto & aa=a[y*2+l];const auto & b0=b[x*2+l];const auto & b1=b[(x+1)*2+l];
                int32x4_t z00=vdupq_n_s32(0),z01=vdupq_n_s32(0),z02=vdupq_n_s32(0),z03=vdupq_n_s32(0);
                int32x4_t z10=vdupq_n_s32(0),z11=vdupq_n_s32(0),z12=vdupq_n_s32(0),z13=vdupq_n_s32(0);
                for(int k=0;k<8;++k) {
                    const int8x16_t av=vld1q_s8(aa.qs+16*k),bv0=vld1q_s8(b0.qs+16*k),bv1=vld1q_s8(b1.qs+16*k);
                    z00=vdotq_laneq_s32(z00,bv0,av,0);z01=vdotq_laneq_s32(z01,bv0,av,1);
                    z02=vdotq_laneq_s32(z02,bv0,av,2);z03=vdotq_laneq_s32(z03,bv0,av,3);
                    z10=vdotq_laneq_s32(z10,bv1,av,0);z11=vdotq_laneq_s32(z11,bv1,av,1);
                    z12=vdotq_laneq_s32(z12,bv1,av,2);z13=vdotq_laneq_s32(z13,bv1,av,3);
                }
                const float32x4_t ad=vcvt_f32_f16(vld1_f16((const float16_t *)aa.d));
                const float32x4_t bd0=vcvt_f32_f16(vld1_f16((const float16_t *)b0.d));
                const float32x4_t bd1=vcvt_f32_f16(vld1_f16((const float16_t *)b1.d));
                // Keep separate scale/product/sum operations; contraction is disabled
                // for this helper without changing other operators.
                f00=vaddq_f32(f00,vmulq_f32(vmulq_laneq_f32(bd0,ad,0),vcvtq_f32_s32(z00)));
                f01=vaddq_f32(f01,vmulq_f32(vmulq_laneq_f32(bd0,ad,1),vcvtq_f32_s32(z01)));
                f02=vaddq_f32(f02,vmulq_f32(vmulq_laneq_f32(bd0,ad,2),vcvtq_f32_s32(z02)));
                f03=vaddq_f32(f03,vmulq_f32(vmulq_laneq_f32(bd0,ad,3),vcvtq_f32_s32(z03)));
                f10=vaddq_f32(f10,vmulq_f32(vmulq_laneq_f32(bd1,ad,0),vcvtq_f32_s32(z10)));
                f11=vaddq_f32(f11,vmulq_f32(vmulq_laneq_f32(bd1,ad,1),vcvtq_f32_s32(z11)));
                f12=vaddq_f32(f12,vmulq_f32(vmulq_laneq_f32(bd1,ad,2),vcvtq_f32_s32(z12)));
                f13=vaddq_f32(f13,vmulq_f32(vmulq_laneq_f32(bd1,ad,3),vcvtq_f32_s32(z13)));
            }
            vst1q_f32(output+(4*y+0)*64+4*x,vaddq_f32(vld1q_f32(output+(4*y+0)*64+4*x),f00));vst1q_f32(output+(4*y+1)*64+4*x,vaddq_f32(vld1q_f32(output+(4*y+1)*64+4*x),f01));
            vst1q_f32(output+(4*y+2)*64+4*x,vaddq_f32(vld1q_f32(output+(4*y+2)*64+4*x),f02));vst1q_f32(output+(4*y+3)*64+4*x,vaddq_f32(vld1q_f32(output+(4*y+3)*64+4*x),f03));
            vst1q_f32(output+(4*y+0)*64+4*x+4,vaddq_f32(vld1q_f32(output+(4*y+0)*64+4*x+4),f10));vst1q_f32(output+(4*y+1)*64+4*x+4,vaddq_f32(vld1q_f32(output+(4*y+1)*64+4*x+4),f11));
            vst1q_f32(output+(4*y+2)*64+4*x+4,vaddq_f32(vld1q_f32(output+(4*y+2)*64+4*x+4),f12));vst1q_f32(output+(4*y+3)*64+4*x+4,vaddq_f32(vld1q_f32(output+(4*y+3)*64+4*x+4),f13));
        }
        return true;
    }
#endif
    GGML_UNUSED(output);GGML_UNUSED(vx);GGML_UNUSED(vy);
    return false;
}
