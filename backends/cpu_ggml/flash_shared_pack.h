#pragma once
#include "common.h"
#include "ops.h"

// Full F16 tiles only: original partial/masked/reference paths remain intact.
// This is transient operator scratch, fully overwritten on every invocation.
static inline bool ggml_fa_shared_kv_eligible(const struct ggml_tensor * dst) {
    const struct ggml_tensor * q=dst->src[0], * k=dst->src[1], * v=dst->src[2];
    return q->type==GGML_TYPE_F32 && k->type==GGML_TYPE_F16 && v->type==GGML_TYPE_F16 &&
           q->ne[0]==64 && k->ne[0]==64 && v->ne[0]==64 && q->ne[1]>=GGML_FA_TILE_Q &&
           k->ne[1]==v->ne[1] && k->ne[1]%GGML_FA_TILE_KV==0 && dst->src[3]==NULL;
}
static inline size_t ggml_fa_shared_kv_plane_floats(const struct ggml_tensor * tensor) {
    return (size_t)tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}
static inline size_t ggml_fa_shared_kv_floats(const struct ggml_tensor * dst) {
    if(!ggml_fa_shared_kv_eligible(dst)) return 0;
    return GGML_PAD(ggml_fa_shared_kv_plane_floats(dst->src[1])+
                    ggml_fa_shared_kv_plane_floats(dst->src[2]),CACHE_LINE_SIZE_F32);
}
