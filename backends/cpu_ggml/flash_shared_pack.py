"""Physical CPU overlay: unpack full unmasked D64 F16 K/V once per flash call.

Pinned upstream GGML arithmetic remains unchanged. The existing pre-query barrier
also publishes the operator-local shared pack. No state/cache/native ABI changes.
"""
from pathlib import Path
import argparse, hashlib


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('upstream source anchor differs: ' + old[:100])
    return text.replace(old, new, 1)


PACK_FUNCTION = r'''
// The planner reserves this shared prefix in the same operator workspace.
// Every element is rewritten before the existing chunk-dispatch barrier.
static void ggml_flash_shared_kv_pack(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * k=dst->src[1], * v=dst->src[2];
    const int64_t tiles=k->ne[1]/GGML_FA_TILE_KV;
    const int64_t k_tiles=tiles*k->ne[2]*k->ne[3];
    const int64_t v_tiles=tiles*v->ne[2]*v->ne[3];
    float * packed=(float *)params->wdata;
    GGML_ASSERT(ggml_fa_shared_kv_floats(dst)*sizeof(float)<=params->wsize);
    for(int64_t index=params->ith;index<k_tiles+v_tiles;index+=params->nth) {
        const bool is_k=index<k_tiles;
        const auto * tensor=is_k?k:v;
        const int64_t local=is_k?index:index-k_tiles;
        const int64_t plane=local/tiles, tile=local%tiles;
        const int64_t head=plane%tensor->ne[2], batch=plane/tensor->ne[2];
        float * out=packed+(is_k?0:ggml_fa_shared_kv_plane_floats(k))+local*GGML_FA_TILE_KV*64;
        for(int tk=0;tk<GGML_FA_TILE_KV;++tk) {
            const auto * row=(const ggml_fp16_t *)((const char *)tensor->data+
                (tile*GGML_FA_TILE_KV+tk)*tensor->nb[1]+head*tensor->nb[2]+batch*tensor->nb[3]);
            if(is_k) {
                for(int dk=0;dk<64;++dk)out[dk*GGML_FA_TILE_KV+tk]=GGML_CPU_FP16_TO_FP32(row[dk]);
            } else {
                ggml_cpu_fp16_to_fp32(row,out+tk*64,64);
            }
        }
    }
}

'''


def ops_overlay(text):
    text = replace_once(text, '#include "simd-gemm.h"\n',
                        '#include "simd-gemm.h"\n#include "flash_shared_pack.h"\n')
    text = replace_once(text, 'static void ggml_compute_forward_flash_attn_ext_tiled(\n',
                        PACK_FUNCTION + 'static void ggml_compute_forward_flash_attn_ext_tiled(\n')
    old = '        float * base  = (float *) params->wdata + ith*(Q_TILE_SZ*DK + 2*Q_TILE_SZ*KV_TILE_SZ + Q_TILE_SZ*DV + KV_TILE_SZ*DV + KV_TILE_SZ*DK + CACHE_LINE_SIZE_F32);'
    new = '''        const bool shared_pack = !params->use_ref && ggml_fa_shared_kv_eligible(dst);
        const size_t shared_floats = shared_pack ? ggml_fa_shared_kv_floats(dst) : 0;
        const float * packed_k = (const float *)params->wdata;
        const float * packed_v = shared_pack ? packed_k + ggml_fa_shared_kv_plane_floats(k) : nullptr;
        float * base  = (float *) params->wdata + shared_floats + ith*(Q_TILE_SZ*DK + 2*Q_TILE_SZ*KV_TILE_SZ + Q_TILE_SZ*DV + KV_TILE_SZ*DV + KV_TILE_SZ*DK + CACHE_LINE_SIZE_F32);'''
    text = replace_once(text, old, new)
    text = replace_once(text, '''        memset(K_f32, 0, DK * KV_TILE_SZ * sizeof(float));
        memset(V32,   0, KV_TILE_SZ * DV * sizeof(float));''', '''        if (!shared_pack) {
            memset(K_f32, 0, DK * KV_TILE_SZ * sizeof(float));
            memset(V32,   0, KV_TILE_SZ * DV * sizeof(float));
        }''')
    start = text.index('            // Pack K tile transposed:')
    end = text.index('            memset(KQ, 0,', start)
    original = text[start:end]
    text = text[:start] + '''            const float * K_gemm = K_f32;
            if (shared_pack) {
                K_gemm = packed_k + ((ik3*nek2+ik2)*(nek1/KV_TILE_SZ)+ic/KV_TILE_SZ)*DK*KV_TILE_SZ;
            } else {
''' + original + '''            }
''' + text[end:]
    text = replace_once(text, 'simd_gemm(KQ, (const float *)Q_q, K_f32, Q_TILE_SZ, DK, KV_TILE_SZ);',
                        'simd_gemm(KQ, (const float *)Q_q, K_gemm, Q_TILE_SZ, DK, KV_TILE_SZ);')
    start = text.index('            // V accumulation: VKQ32 += softmax(KQ) * V')
    end = text.index('            for (int tq = 0; tq < Q_TILE_SZ; tq++) {\n                if (skip[tq])', start)
    original = text[start:end]
    text = text[:start] + '''            const float * V_gemm = V32;
            if (shared_pack) {
                V_gemm = packed_v + ((iv3*nev2+iv2)*(nev1/KV_TILE_SZ)+ic/KV_TILE_SZ)*DV*KV_TILE_SZ;
            } else {
''' + original + '''            }
''' + text[end:]
    text = replace_once(text, 'simd_gemm(VKQ32, KQ, V32, Q_TILE_SZ, KV_TILE_SZ, DV);',
                        'simd_gemm(VKQ32, KQ, V_gemm, Q_TILE_SZ, KV_TILE_SZ, DV);')
    # Move eligibility and the pack before the ORIGINAL initial barrier. This
    # preserves one barrier and publishes both packed planes and chunk counter.
    start = text.index('        if (ith == 0) {\n            ggml_threadpool_chunk_set(params->threadpool, nth);')
    end = text.index('        int current_chunk = ith;', start)
    old = text[start:end]
    eligibility = old[old.index('        static constexpr int64_t Q_TILE_SZ'):]
    barrier = old[:old.index('        const int64_t dr =')]
    replacement = eligibility + '''        if (use_tiled && ggml_fa_shared_kv_eligible(dst)) {
            ggml_flash_shared_kv_pack(params, dst);
        }
''' + barrier + '''        const int64_t dr = (nr + nchunk - 1) / nchunk;

'''
    text = text[:start] + replacement + text[end:]
    return text


def planner_overlay(text):
    text = replace_once(text, '#include "common.h"\n', '#include "common.h"\n#include "flash_shared_pack.h"\n')
    return replace_once(text, '                        cur += MAX(prefill, decode);',
                        '                        cur += MAX(prefill, decode) + sizeof(float)*ggml_fa_shared_kv_floats(node);')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ggml-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cpu = args.ggml_source / 'src/ggml-cpu'
    for name, function in [('ops.cpp', ops_overlay), ('ggml-cpu.c', planner_overlay)]:
        text = function((cpu/name).read_text())
        (args.output/name).write_text(text)
        print(name, hashlib.sha256(text.encode()).hexdigest())
    (args.output/'flash_shared_pack.h').write_bytes(Path(__file__).with_suffix('.h').read_bytes())


if __name__ == '__main__':
    main()
