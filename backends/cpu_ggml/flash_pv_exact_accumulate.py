"""Encoder-only original-rounding direct PV accumulation with invocation-local finite history."""
from flash_shared_pack import replace_once


def ops_overlay(text):
    text=replace_once(text, '#include "flash_pv_sdot.h"', '#include "flash_pv_sdot.h"\n#include "flash_pv_exact_accumulate.h"')
    text=replace_once(text, '        memset(VKQ32, 0, Q_TILE_SZ * DV * sizeof(float));', "        // This flag belongs only to this Q tile's existing zeroed accumulator.\n        // A single original F32 PV fallback invalidates it for ALL later K tiles:\n        // a later finite score/value tile cannot prove prior VKQ finite.\n        bool pv_exact_finite_history = true;\n        memset(VKQ32, 0, Q_TILE_SZ * DV * sizeof(float));")
    text=replace_once(text, '                mstar_pv_gemm4x8(64,KQ,64,pv_values,V32,64,64);\n                ggml_vec_acc_f32(64*64,VKQ32,KQ);', '                // Existing guards establish finite S/M (therefore finite\n                // nonnegative exp probabilities <=2), and the finite-V sentinel\n                // proves all original F16 values <=65504. The current Q8 packer\n                // bounds int8 dots; at most1536 keys and rescale factors in[0,1]\n                // keep zero-origin finite history far below F32 overflow.\n                // All other geometries/paths retain the original arithmetic.\n                if (!mstar_pv_exact_accumulate64(VKQ32,pv_values,V32,\n                        pv_exact_finite_history && nek1<=1536)) {\n                    mstar_pv_gemm4x8(64,KQ,64,pv_values,V32,64,64);\n                    ggml_vec_acc_f32(64*64,VKQ32,KQ);\n                }')
    text=replace_once(text, '            } else {\n                simd_gemm(VKQ32, KQ, V_gemm, Q_TILE_SZ, KV_TILE_SZ, DV);\n            }', '            } else {\n                pv_exact_finite_history = false;\n                simd_gemm(VKQ32, KQ, V_gemm, Q_TILE_SZ, KV_TILE_SZ, DV);\n            }')
    return text
