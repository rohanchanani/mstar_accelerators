#pragma once

// Metadata/ABI only. Common provider integration remains the native owner's
// responsibility: actual CUSTOM identity, hardware checks, descriptor ranges,
// producer dependencies, output non-aliasing, canonical arena and retirement.
#include "ggml.h"
#include <cmath>
#include <cstdint>
#include <initializer_list>

namespace managed_sdpa_decode {
constexpr uint32_t head_dim = 64;
constexpr uint32_t max_tokens = 1536;
constexpr uint32_t workgroup = 256;
constexpr uint32_t subgroup = 16;
constexpr uint32_t shared_bytes = (max_tokens + workgroup + subgroup * head_dim) * sizeof(float);
static_assert(shared_bytes == 11264);

struct Parameters {
    uint32_t heads, tokens, batches;
    uint32_t q_head_stride, q_batch_stride;
    uint32_t k_head_stride, k_token_stride, k_batch_stride;
    uint32_t v_head_stride, v_token_stride, v_batch_stride;
    uint32_t mask_batch_stride;
    uint32_t out_head_stride, out_batch_stride;
    float scale;
};
static_assert(sizeof(Parameters) == 60);

inline bool addressable_f32(const ggml_tensor * t) {
    if (!t || t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float)) return false;
    uint64_t last = 0, elements = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] <= 0 || uint64_t(t->ne[i]) > UINT32_MAX ||
            t->nb[i] % sizeof(float) || t->nb[i] > UINT32_MAX) return false;
        if (elements > UINT32_MAX / uint64_t(t->ne[i])) return false;
        elements *= uint64_t(t->ne[i]);
        const uint64_t extent = uint64_t(t->ne[i] - 1) * (t->nb[i] / sizeof(float));
        if (extent > UINT32_MAX - last) return false;
        last += extent;
    }
    // Vulkan maxStorageBufferRange is a uint32 byte extent; the common
    // provider additionally enforces the actual (usually smaller) device cap.
    return last < UINT32_MAX / sizeof(float);
}

inline bool metadata(const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v,
                     const ggml_tensor * mask, const ggml_tensor * out,
                     float scale, Parameters & parameters) {
    if (!addressable_f32(q) || !addressable_f32(k) || !addressable_f32(v) ||
        !addressable_f32(mask) || !addressable_f32(out) || !std::isfinite(scale)) return false;
    if (q->ne[0] != head_dim || q->ne[2] != 1 || q->ne[3] > 2) return false;
    const uint64_t h = uint64_t(q->ne[1]), b = uint64_t(q->ne[3]), t = uint64_t(k->ne[2]);
    if (!t || t > max_tokens) return false;
    for (const auto * x : {k, v}) {
        if (x->ne[0] != head_dim || uint64_t(x->ne[1]) != h ||
            uint64_t(x->ne[2]) != t || uint64_t(x->ne[3]) != b ||
            x->nb[1] < head_dim * sizeof(float)) return false;
        const uint64_t head_span = (h - 1) * x->nb[1] + head_dim * sizeof(float);
        if (x->nb[2] < head_span || (b > 1 && x->nb[3] < (t - 1) * x->nb[2] + head_span)) return false;
    }
    const uint64_t query_span = (h - 1) * q->nb[1] + head_dim * sizeof(float);
    if (q->nb[1] < head_dim * sizeof(float) || (b > 1 && q->nb[3] < query_span)) return false;
    if (uint64_t(mask->ne[0]) != t || mask->ne[1] != 1 || mask->ne[2] != 1 ||
        (mask->ne[3] != 1 && uint64_t(mask->ne[3]) != b) ||
        (mask->ne[3] > 1 && mask->nb[3] < t * sizeof(float))) return false;
    if (out->view_src || out->ne[0] != head_dim || uint64_t(out->ne[1]) != h ||
        out->ne[2] != 1 || uint64_t(out->ne[3]) != b ||
        out->nb[1] != head_dim * sizeof(float) ||
        out->nb[2] != h * head_dim * sizeof(float) ||
        out->nb[3] != h * head_dim * sizeof(float)) return false;
    parameters = {uint32_t(h), uint32_t(t), uint32_t(b),
        uint32_t(q->nb[1] / 4), uint32_t(q->nb[3] / 4),
        uint32_t(k->nb[1] / 4), uint32_t(k->nb[2] / 4), uint32_t(k->nb[3] / 4),
        uint32_t(v->nb[1] / 4), uint32_t(v->nb[2] / 4), uint32_t(v->nb[3] / 4),
        mask->ne[3] == 1 ? 0u : uint32_t(mask->nb[3] / 4),
        uint32_t(out->nb[1] / 4), uint32_t(out->nb[3] / 4), scale};
    return true;
}
} // namespace managed_sdpa_decode
