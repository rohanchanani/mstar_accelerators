"""Encoder PV-only 4x8 physical tile; canonical packing and scratch unchanged."""

def ops_overlay(text):
    include = '#include "flash_pv_sdot.h"'
    original = 'ggml_gemm_q8_0_4x4_q8_0(64,KQ,64,pv_values,V32,64,64);'
    if text.count(include) != 1 or text.count(original) != 1:
        raise ValueError('PV GEMM overlay requires the exact admitted PV SDOT source')
    return text.replace(include, include + '\n#include "flash_pv_gemm.h"').replace(original, 'mstar_pv_gemm4x8(64,KQ,64,pv_values,V32,64,64);')
