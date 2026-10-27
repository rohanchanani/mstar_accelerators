"""Encoder-marked QK-only SDOT; keep tiled F32 softmax/PV and admitted scratch."""
from flash_shared_pack import replace_once

PACK = r'''
// The original shared F32 K plane is larger than the packed Q8 representation.
// Keep its full planner allocation and the V plane's original offset. No new
// allocation, hidden cache, extra barrier, or workspace lifetime is introduced.
static bool ggml_flash_qk_sdot(const ggml_tensor * dst) {
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int32_t tag, precision;
    memcpy(&tag, (const char *)dst->op_params + 4*sizeof(int32_t), sizeof(tag));
    memcpy(&precision, (const char *)dst->op_params + 3*sizeof(int32_t), sizeof(precision));
    const auto * q=dst->src[0], * k=dst->src[1], * v=dst->src[2];
    return tag == MSTAR_ENCODER_QK_SDOT_TAG && precision == GGML_PREC_DEFAULT &&
        ggml_fa_shared_kv_eligible(dst) && q->nb[0]==sizeof(float) &&
        k->nb[0]==sizeof(ggml_fp16_t) && v->nb[0]==sizeof(ggml_fp16_t) &&
        k->ne[2]>0 && k->ne[3]>0 && v->ne[2]>0 && v->ne[3]>0 &&
        q->ne[2]%k->ne[2]==0 && q->ne[3]%k->ne[3]==0 &&
        q->ne[2]%v->ne[2]==0 && q->ne[3]%v->ne[3]==0 &&
        ggml_fa_tile_config::Q == 64 && ggml_fa_tile_config::KV == 64;
#else
    GGML_UNUSED(dst); return false;
#endif
}
static constexpr size_t ggml_flash_qk_row_bytes = 2*sizeof(block_q8_0x4)/4;
static_assert(sizeof(block_q8_0x4)==136, "pinned 4-row Q8 block ABI differs");
static_assert(ggml_flash_qk_row_bytes*64 <= 64*64*sizeof(float), "Q pack exceeds existing K tile scratch");

'''

def ops_overlay(text):
    text = replace_once(text, '#include "flash_shared_pack.h"\n',
        '#include "flash_shared_pack.h"\n#include "flash_qk_sdot.h"\n#include "repack.h"\n')
    text = replace_once(text, '// The planner reserves this shared prefix in the same operator workspace.',
        PACK + '// The planner reserves this shared prefix in the same operator workspace.')
    text = replace_once(text, '    float * packed=(float *)params->wdata;',
        '    const bool qk_sdot = ggml_flash_qk_sdot(dst);\n    GGML_ASSERT((uintptr_t)params->wdata % alignof(float) == 0);\n    GGML_ASSERT(ggml_fa_shared_kv_plane_floats(k)/64*ggml_flash_qk_row_bytes <= ggml_fa_shared_kv_plane_floats(k)*sizeof(float));\n    float * packed=(float *)params->wdata;')
    text = replace_once(text, '''        for(int tk=0;tk<GGML_FA_TILE_KV;++tk) {''', '''        if (is_k && qk_sdot) {
            float rows[4*64];
            auto * quantized = (char *)params->wdata + local*GGML_FA_TILE_KV*ggml_flash_qk_row_bytes;
            for (int tk=0;tk<GGML_FA_TILE_KV;tk+=4) {
                for (int r=0;r<4;++r) {
                    const auto * row=(const ggml_fp16_t *)((const char *)tensor->data+
                        (tile*GGML_FA_TILE_KV+tk+r)*tensor->nb[1]+head*tensor->nb[2]+batch*tensor->nb[3]);
                    ggml_cpu_fp16_to_fp32(row, rows+r*64, 64);
                }
                ggml_quantize_mat_q8_0_4x4(rows, quantized+tk*ggml_flash_qk_row_bytes, 64);
            }
            continue;
        }
        for(int tk=0;tk<GGML_FA_TILE_KV;++tk) {''')
    text = replace_once(text, '        const bool shared_pack = !params->use_ref && ggml_fa_shared_kv_eligible(dst);',
        '        const bool shared_pack = !params->use_ref && ggml_fa_shared_kv_eligible(dst);\n        const bool qk_sdot = shared_pack && ggml_flash_qk_sdot(dst);')
    text = replace_once(text, '''        if (!shared_pack) {
            memset(K_f32''', '''        if (qk_sdot) {
            GGML_ASSERT((uintptr_t)K_f32 % alignof(block_q8_0x4) == 0);
            GGML_ASSERT((char *)(K_f32+KV_TILE_SZ*DK) <= (char *)params->wdata+params->wsize);
            // K_f32 is unused by shared packing; its 16 KiB holds this 4352 B
            // packed Q tile, rewritten for every (including partial) Q chunk.
            for (int tq=0;tq<Q_TILE_SZ;tq+=4) {
                ggml_quantize_mat_q8_0_4x4((const float *)Q_q+tq*DK,
                    (char *)K_f32+tq*ggml_flash_qk_row_bytes, DK);
            }
        }
        if (!shared_pack) {
            memset(K_f32''')
    text = replace_once(text, '            simd_gemm(KQ, (const float *)Q_q, K_gemm, Q_TILE_SZ, DK, KV_TILE_SZ);', '''            if (qk_sdot) {
                const size_t tile=((ik3*nek2+ik2)*(nek1/KV_TILE_SZ)+ic/KV_TILE_SZ);
                const void * keys=(const char *)packed_k+tile*KV_TILE_SZ*ggml_flash_qk_row_bytes;
                ggml_gemm_q8_0_4x4_q8_0(DK, KQ, KV_TILE_SZ, keys, K_f32, Q_TILE_SZ, KV_TILE_SZ);
            } else {
                simd_gemm(KQ, (const float *)Q_q, K_gemm, Q_TILE_SZ, DK, KV_TILE_SZ);
            }''')
    return text
