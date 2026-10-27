"""Encoder-marked transient Q8 PV, using only existing admitted operator scratch."""
from flash_shared_pack import replace_once

ELIGIBILITY = r'''
static bool ggml_flash_pv_sdot(const ggml_tensor * dst) {
    int32_t tag;
    memcpy(&tag, (const char *)dst->op_params+5*sizeof(tag), sizeof(tag));
    if (tag != MSTAR_ENCODER_PV_SDOT_TAG || !ggml_flash_qk_sdot(dst)) return false;
    const size_t krows=ggml_fa_shared_kv_plane_floats(dst->src[1])/64;
    const size_t vrows=ggml_fa_shared_kv_plane_floats(dst->src[2])/64;
    // Kq remains [0,krows*68); Vq follows it, both strictly within the old
    // F32 K prefix. Keep the original F32 V plane at its unchanged offset.
    const size_t capacity=ggml_fa_shared_kv_plane_floats(dst->src[1])*sizeof(float);
    const size_t kbytes=krows*ggml_flash_qk_row_bytes;
    return kbytes<=capacity && vrows<=(capacity-kbytes)/ggml_flash_qk_row_bytes;
}
static_assert(64*ggml_flash_qk_row_bytes <= 64*64*sizeof(float), "Pq must fit unused V32");

'''

def ops_overlay(text):
    text=replace_once(text,'#include "flash_qk_sdot.h"\n','#include "flash_qk_sdot.h"\n#include "flash_pv_sdot.h"\n')
    text=replace_once(text,'// The planner reserves this shared prefix in the same operator workspace.',ELIGIBILITY+'// The planner reserves this shared prefix in the same operator workspace.')
    text=replace_once(text,'    const bool qk_sdot = ggml_flash_qk_sdot(dst);','    const bool qk_sdot = ggml_flash_qk_sdot(dst);\n    const bool pv_sdot = ggml_flash_pv_sdot(dst);')
    anchor='''        }
    }
}

static void ggml_compute_forward_flash_attn_ext_tiled('''
    text=replace_once(text,anchor,'''        }
        if (!is_k && pv_sdot) {
            // This task has just filled its complete F32 V tile; no other task
            // reads it until the ORIGINAL shared-pack publication barrier.
            float rows[4*64];
            const size_t kbytes=ggml_fa_shared_kv_plane_floats(k)/64*ggml_flash_qk_row_bytes;
            auto * quantized=(char *)params->wdata+kbytes+local*64*ggml_flash_qk_row_bytes;
            GGML_ASSERT(quantized+64*ggml_flash_qk_row_bytes <=
                        (char *)params->wdata+ggml_fa_shared_kv_plane_floats(k)*sizeof(float));
            bool finite=true;
            for(int i=0;i<64*64;++i)finite=finite && std::isfinite(out[i]);
            if(!finite) {
                // Finite F16 V can never produce a NaN Q8 scale. Preserve the
                // original F32 V and mark only this transient quantized tile.
                memset(quantized,0,64*ggml_flash_qk_row_bytes);
                ((block_q8_0x4 *)quantized)->d[0]=GGML_CPU_FP32_TO_FP16(NAN);
                continue;
            }
            for(int ch=0;ch<64;ch+=4) {
                for(int r=0;r<4;++r)for(int key=0;key<64;++key)rows[r*64+key]=out[key*64+ch+r];
                ggml_quantize_mat_q8_0_4x4(rows,quantized+ch*ggml_flash_qk_row_bytes,64);
            }
        }
    }
}

static void ggml_compute_forward_flash_attn_ext_tiled(''')
    text=replace_once(text,'        const bool qk_sdot = shared_pack && ggml_flash_qk_sdot(dst);','        const bool qk_sdot = shared_pack && ggml_flash_qk_sdot(dst);\n        const bool pv_sdot = qk_sdot && ggml_flash_pv_sdot(dst);')
    text=replace_once(text,'            simd_gemm(VKQ32, KQ, V_gemm, Q_TILE_SZ, KV_TILE_SZ, DV);',r'''            const size_t pv_tile=((iv3*nev2+iv2)*(nev1/KV_TILE_SZ)+ic/KV_TILE_SZ);
            const size_t pv_kbytes=ggml_fa_shared_kv_plane_floats(k)/64*ggml_flash_qk_row_bytes;
            const auto * pv_values=pv_sdot ? (const block_q8_0x4 *)((const char *)params->wdata+pv_kbytes+pv_tile*64*ggml_flash_qk_row_bytes) : nullptr;
            bool pv_finite=true;
            if(pv_sdot)for(int row=0;row<64;++row)
                pv_finite=pv_finite && std::isfinite(S[row]) && (std::isfinite(M[row]) || skip[row]);
            if (pv_sdot && pv_finite && std::isfinite(GGML_CPU_FP16_TO_FP32(pv_values->d[0]))) {
                // K_f32 retains the packed Q for the NEXT QK. V32 is otherwise
                // unused with shared packing: use it only for this Pq tile.
                GGML_ASSERT((char *)(V32+KV_TILE_SZ*DV) <= (char *)K_f32);
                for(int row=0;row<64;row+=4)
                    ggml_quantize_mat_q8_0_4x4(KQ+row*64,(char *)V32+row*ggml_flash_qk_row_bytes,64);
                // All S/M and skip consumers finished; KQ can now be overwritten
                // by the scaled integer-dot result before FP32 accumulation.
                ggml_gemm_q8_0_4x4_q8_0(64,KQ,64,pv_values,V32,64,64);
                ggml_vec_acc_f32(64*64,VKQ32,KQ);
            } else {
                simd_gemm(VKQ32, KQ, V_gemm, Q_TILE_SZ, KV_TILE_SZ, DV);
            }''')
    return text
