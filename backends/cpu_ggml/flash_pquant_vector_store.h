#pragma once
// Exact encoder probability packing: retain staging, reduction and scale order.
// Non-saturating narrowing preserves the low byte of each converted int32,
// matching scalar int8 stores while writing the same packed row order.
#include "repack.h"
#include "vec.h"
static inline void mstar_pquant_vector_store64(const float * GGML_RESTRICT x,void * GGML_RESTRICT output) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    auto *y=(block_q8_0x4 *)output;
    float32x4_t srcv[4][8];float id[4];
    for(int block=0;block<2;++block){
        float32x4_t asrcv[8],amaxv[8];
        for(int row=0;row<4;++row){
            for(int j=0;j<8;++j)srcv[row][j]=vld1q_f32(x+row*64+block*32+4*j);
            for(int j=0;j<8;++j)asrcv[j]=vabsq_f32(srcv[row][j]);
            for(int j=0;j<4;++j)amaxv[2*j]=vmaxq_f32(asrcv[2*j],asrcv[2*j+1]);
            for(int j=0;j<2;++j)amaxv[4*j]=vmaxq_f32(amaxv[4*j],amaxv[4*j+2]);
            for(int j=0;j<1;++j)amaxv[8*j]=vmaxq_f32(amaxv[8*j],amaxv[8*j+4]);
            const float amax=vmaxvq_f32(amaxv[0]);
            const float d=amax/127.f;id[row]=d?1.f/d:0.f;
            y[block].d[row]=GGML_CPU_FP32_TO_FP16(d);
        }
        for(int j=0;j<8;++j){
            const auto a=vcvtnq_s32_f32(vmulq_n_f32(srcv[0][j],id[0]));
            const auto b=vcvtnq_s32_f32(vmulq_n_f32(srcv[1][j],id[1]));
            const auto c=vcvtnq_s32_f32(vmulq_n_f32(srcv[2][j],id[2]));
            const auto d=vcvtnq_s32_f32(vmulq_n_f32(srcv[3][j],id[3]));
            const auto ab=vcombine_s16(vmovn_s32(a),vmovn_s32(b));
            const auto cd=vcombine_s16(vmovn_s32(c),vmovn_s32(d));
            const auto bytes=vcombine_s8(vmovn_s16(ab),vmovn_s16(cd));
            vst1q_s8(y[block].qs+j*16,bytes);
        }
    }
#else
    ggml_quantize_mat_q8_0_4x4(x,output,64);
#endif
}
