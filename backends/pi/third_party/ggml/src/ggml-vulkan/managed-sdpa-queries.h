#pragma once

// New bounded multi-query ABI. The accepted Q1 ABI/shader remain unchanged.
// Common integration owns CUSTOM identity, current-device capability checks,
// exact canonical descriptor ranges, producer edges, output non-overlap,
// range barriers and normal arena/submission/retirement ownership.
#include "managed-sdpa-decode.h"

namespace managed_sdpa_queries {
constexpr uint32_t head_dim = 64;
constexpr uint32_t max_tokens = 1536;
constexpr uint32_t max_queries = 4;
constexpr uint32_t workgroup = 256;
constexpr uint32_t subgroup = 16;
constexpr uint32_t shared_bytes = managed_sdpa_decode::shared_bytes;

// All strides are F32 element counts. Dispatch is (heads,batches,queries),
// one independent WG per query: selected device must admit every grid axis.
struct Parameters {
    uint32_t heads, tokens, queries, batches;
    uint32_t q_head_stride, q_query_stride, q_batch_stride;
    uint32_t k_head_stride, k_token_stride, k_batch_stride;
    uint32_t v_head_stride, v_token_stride, v_batch_stride;
    uint32_t mask_query_stride, mask_batch_stride;
    uint32_t out_head_stride, out_query_stride, out_batch_stride;
    float scale;
};
static_assert(sizeof(Parameters) == 76);
static_assert(shared_bytes == 11264);

inline bool metadata(const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v,
                     const ggml_tensor * mask, const ggml_tensor * out,
                     float scale, Parameters & parameters) {
    using managed_sdpa_decode::addressable_f32;
    if (!addressable_f32(q) || !addressable_f32(k) || !addressable_f32(v) ||
        !addressable_f32(mask) || !addressable_f32(out) || !std::isfinite(scale)) return false;
    if (q->ne[0] != head_dim || q->ne[2] > max_queries || q->ne[3] > 2) return false;
    const uint64_t h = uint64_t(q->ne[1]), n = uint64_t(q->ne[2]);
    const uint64_t b = uint64_t(q->ne[3]), t = uint64_t(k->ne[2]);
    if (!t || t > max_tokens) return false;
    for (const auto * x : {k, v}) {
        if (x->ne[0] != head_dim || uint64_t(x->ne[1]) != h ||
            uint64_t(x->ne[2]) != t || uint64_t(x->ne[3]) != b ||
            x->nb[1] < head_dim * sizeof(float)) return false;
        const uint64_t head_span = (h - 1) * x->nb[1] + head_dim * sizeof(float);
        if (x->nb[2] < head_span || (b > 1 && x->nb[3] < (t - 1) * x->nb[2] + head_span)) return false;
    }
    const uint64_t query_span = (h - 1) * q->nb[1] + head_dim * sizeof(float);
    if (q->nb[1] < head_dim * sizeof(float) || (n > 1 && q->nb[2] < query_span) ||
        (b > 1 && q->nb[3] < (n - 1) * q->nb[2] + query_span)) return false;
    if (uint64_t(mask->ne[0]) != t || (mask->ne[1] != 1 && uint64_t(mask->ne[1]) != n) ||
        mask->ne[2] != 1 || (mask->ne[3] != 1 && uint64_t(mask->ne[3]) != b) ||
        (mask->ne[1] > 1 && mask->nb[1] < t * sizeof(float)) ||
        (mask->ne[3] > 1 && mask->nb[3] < (uint64_t(mask->ne[1]) - 1) * mask->nb[1] + t * sizeof(float))) return false;
    if (out->view_src || out->ne[0] != head_dim || uint64_t(out->ne[1]) != h ||
        uint64_t(out->ne[2]) != n || uint64_t(out->ne[3]) != b ||
        out->nb[1] != head_dim * sizeof(float) ||
        out->nb[2] != h * head_dim * sizeof(float) ||
        out->nb[3] != n * h * head_dim * sizeof(float)) return false;
    parameters = {uint32_t(h), uint32_t(t), uint32_t(n), uint32_t(b),
        uint32_t(q->nb[1] / 4), uint32_t(q->nb[2] / 4), uint32_t(q->nb[3] / 4),
        uint32_t(k->nb[1] / 4), uint32_t(k->nb[2] / 4), uint32_t(k->nb[3] / 4),
        uint32_t(v->nb[1] / 4), uint32_t(v->nb[2] / 4), uint32_t(v->nb[3] / 4),
        mask->ne[1] == 1 ? 0u : uint32_t(mask->nb[1] / 4),
        mask->ne[3] == 1 ? 0u : uint32_t(mask->nb[3] / 4),
        uint32_t(out->nb[1] / 4), uint32_t(out->nb[2] / 4), uint32_t(out->nb[3] / 4), scale};
    return true;
}
} // namespace managed_sdpa_queries
