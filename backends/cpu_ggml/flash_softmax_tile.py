"""Encoder-marked whole-tile softmax; unchanged exceptional/generic primitives."""
from flash_shared_pack import replace_once

def ops_overlay(text):
    text=replace_once(text,'#include "vec.h"\n','#include "vec.h"\n#include "flash_softmax_tile.h"\n')
    start=text.index('            for (int tq = 0; tq < Q_TILE_SZ; tq++) {\n                float * kq_row = KQ + tq * KV_TILE_SZ;')
    end=text.index('            const float * V_gemm = V32;',start)
    original=text[start:end]
    text=text[:start]+'''            if (qk_sdot) {
                mstar_flash_softmax64(KQ, VKQ32, S, M, skip);
            } else {
'''+original+'''            }

'''+text[end:]
    return text
