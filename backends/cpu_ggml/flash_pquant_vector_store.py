"""Exact encoder-P-only packed stores; block32 math/layout/guards unchanged."""

def ops_overlay(text):
    include = '#include "flash_pv_sdot.h"'
    original = 'ggml_quantize_mat_q8_0_4x4(KQ+row*64,(char *)V32+row*ggml_flash_qk_row_bytes,64);'
    if text.count(include) != 1 or text.count(original) != 1:
        raise ValueError('P vector stores require the exact admitted encoder PV producer')
    return text.replace(include, include + '\n#include "flash_pquant_vector_store.h"').replace(
        original, 'mstar_pquant_vector_store64(KQ+row*64,(char *)V32+row*ggml_flash_qk_row_bytes);')
