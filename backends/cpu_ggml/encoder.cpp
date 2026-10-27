// Graph topology follows ggml-org/whisper.cpp (MIT), pinned by build.py.
// Unlike its Whisper context this component owns no decoder, KV or sampler.
// F32 entry retains explicit F32 im2col/KV and erf GELU. The distinct Q8 entry
// copies upstream's Q8_REPACK/F16-conv/F16-attention/tanh-GELU physical math.
#include "encoder.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml-cpu/ops.h"
#include "ggml-cpu/binary-ops.h"
#include "q8_provider.h"
#include "flash_qk_sdot.h"
#include "flash_pv_sdot.h"
#include "encoder_gelu_lut.h"

#include <atomic>
#include <array>
#include <set>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Physical experiment only: bit0=conv1, bit1=conv2. Normal builds retain F16.
#ifndef MSTAR_ENCODER_Q8_CONV
#define MSTAR_ENCODER_Q8_CONV 0
#endif
static_assert(MSTAR_ENCODER_Q8_CONV >= 0 && MSTAR_ENCODER_Q8_CONV <= 3, "invalid Q8 convolution selection");

// 0 preserves the full-query graph. Query tiling is a physical arena-local
// pipeline; every logical row still executes the same norm/FC1/GELU/FC2.
#ifndef MSTAR_ENCODER_FFN_TILE
#define MSTAR_ENCODER_FFN_TILE 0
#endif
static_assert(MSTAR_ENCODER_FFN_TILE == 0 || MSTAR_ENCODER_FFN_TILE == 128,
              "FFN tile must be 0 or 128");

namespace {
ggml_tensor * convolution_q8(ggml_context * ctx, ggml_tensor * kernel,
                             ggml_tensor * quantized, ggml_tensor * input, int stride) {
    auto * columns = ggml_im2col(ctx, kernel, input, stride, 0, 1, 0, 1, 0, false, GGML_TYPE_F32);
    columns = ggml_reshape_2d(ctx, columns, columns->ne[0], columns->ne[1]*columns->ne[2]);
    if (columns->ne[0] < quantized->ne[0])
        columns = ggml_pad(ctx, columns, int(quantized->ne[0] - columns->ne[0]), 0, 0, 0);
    return ggml_mul_mat(ctx, quantized, columns);
}
constexpr size_t max_nodes = 4096;
void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}
void error_text(char * out, size_t n, const char * message) {
    if (out && n) std::snprintf(out, n, "%s", message);
}
bool aligned(const void * p) { return p && uintptr_t(p) % alignof(float) == 0; }
bool disjoint(const void * a, size_t an, const void * b, size_t bn) {
    const uintptr_t aa = uintptr_t(a), bb = uintptr_t(b);
    return aa <= UINTPTR_MAX - an && bb <= UINTPTR_MAX - bn &&
           (aa + an <= bb || bb + bn <= aa);
}

// Same compiled ADD then GELU kernels as separate graph nodes. Complete one
// row before the next, keeping the intermediate in the CUSTOM output cache.
void encoder_bias_gelu_compiled(ggml_tensor * dst, const ggml_tensor * projection,
                       const ggml_tensor * bias, int ith, int nth, void *) {
    const ggml_compute_params single{0,1,0,nullptr,nullptr,false};
    const int64_t begin = dst->ne[1]*ith/nth, end = dst->ne[1]*(ith+1)/nth;
    for (int64_t row = begin; row < end; ++row) {
        ggml_tensor source = *projection, biased = *dst;
        source.data = static_cast<uint8_t *>(projection->data) + row*projection->nb[1];
        biased.data = static_cast<uint8_t *>(dst->data) + row*dst->nb[1];
        source.ne[1] = biased.ne[1] = 1;
        source.nb[2] = source.nb[3] = source.nb[1];
        biased.nb[2] = biased.nb[3] = biased.nb[1];
        biased.op = GGML_OP_ADD; biased.src[0] = &source;
        biased.src[1] = const_cast<ggml_tensor *>(bias);
        ggml_compute_forward_add(&single, &biased);
        auto activated = biased; activated.op = GGML_OP_UNARY;
        activated.src[0] = &biased;
        const int32_t op = GGML_UNARY_OP_GELU;
        std::memcpy(activated.op_params, &op, sizeof(op));
        ggml_compute_forward_unary(&single, &activated);
    }
}

void encoder_bias_gelu(ggml_tensor * dst, const ggml_tensor * projection,
                       const ggml_tensor * bias, int ith, int nth, void * userdata) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (encoder_gelu_lut_native() && dst->type == GGML_TYPE_F32 &&
        projection->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32 &&
        dst->ne[0] % 4 == 0 && projection->ne[0] == dst->ne[0] &&
        projection->ne[1] == dst->ne[1] && dst->ne[2] == 1 && dst->ne[3] == 1 &&
        projection->ne[2] == 1 && projection->ne[3] == 1 &&
        bias->ne[1] == 1 && bias->ne[2] == 1 && bias->ne[3] == 1 &&
        dst->nb[0] == sizeof(float) && projection->nb[0] == sizeof(float) &&
        bias->nb[0] == sizeof(float) && bias->ne[0] == dst->ne[0]) {
        const int64_t begin = dst->ne[1]*ith/nth, end = dst->ne[1]*(ith+1)/nth;
        for (int64_t row = begin; row < end; ++row) {
            auto * result = reinterpret_cast<float *>(static_cast<uint8_t *>(dst->data) + row*dst->nb[1]);
            const auto * source = reinterpret_cast<const float *>(static_cast<const uint8_t *>(projection->data) + row*projection->nb[1]);
            encoder_bias_gelu_lut_row(result, source, static_cast<const float *>(bias->data), int(dst->ne[0]));
        }
        return;
    }
#endif
    encoder_bias_gelu_compiled(dst, projection, bias, ith, nth, userdata);
}

// Same compiled NORM -> MUL -> ADD, completing each row while its intermediate
// remains local. The CUSTOM result owns separate storage: its input also feeds
// the residual edge, so it must never be modified in place.
void encoder_norm_affine(ggml_tensor * dst, const ggml_tensor * input,
                         const ggml_tensor * weight, const ggml_tensor * bias,
                         int ith, int nth, void *) {
    const ggml_compute_params single{0, 1, 0, nullptr, nullptr, false};
    const float epsilon = 1.e-5f;
    const int64_t begin = dst->ne[1]*ith/nth, end = dst->ne[1]*(ith+1)/nth;
    auto source = *input, normalized = *dst;
    source.ne[1] = normalized.ne[1] = 1;
    source.nb[2] = source.nb[3] = source.nb[1];
    normalized.nb[2] = normalized.nb[3] = normalized.nb[1];
    normalized.op = GGML_OP_NORM; normalized.src[0] = &source;
    std::memcpy(normalized.op_params, &epsilon, sizeof(epsilon));
    auto scaled = normalized;
    scaled.op = GGML_OP_MUL; scaled.src[0] = &normalized;
    scaled.src[1] = const_cast<ggml_tensor *>(weight);
    auto shifted = scaled;
    shifted.op = GGML_OP_ADD; shifted.src[0] = &scaled;
    shifted.src[1] = const_cast<ggml_tensor *>(bias);
    for (int64_t row = begin; row < end; ++row) {
        source.data = static_cast<uint8_t *>(input->data) + row*input->nb[1];
        normalized.data = scaled.data = shifted.data = static_cast<uint8_t *>(dst->data) + row*dst->nb[1];
        ggml_compute_forward_norm(&single, &normalized);
        ggml_compute_forward_mul(&single, &scaled);
        ggml_compute_forward_add(&single, &shifted);
    }
}

// Keep the existing projection+bias, then residual+biased operand order.
// The CUSTOM aliases only the sole-consumer projection, never the residual.
void encoder_bias_residual(ggml_tensor * dst, const ggml_tensor * projection,
                           const ggml_tensor * bias, const ggml_tensor * residual,
                           int ith, int nth, void *) {
    const ggml_compute_params single{0,1,0,nullptr,nullptr,false};
    ggml_tensor source = *projection, prior = *residual, biased = *dst, result = *dst;
    for (auto * row : {&source, &prior, &biased, &result}) {
        row->ne[1] = 1;
        row->nb[2] = row->nb[3] = row->nb[1];
    }
    biased.op = GGML_OP_ADD; biased.src[0] = &source;
    biased.src[1] = const_cast<ggml_tensor *>(bias);
    result.op = GGML_OP_ADD; result.src[0] = &prior; result.src[1] = &biased;
    const int64_t begin = dst->ne[1]*ith/nth, end = dst->ne[1]*(ith+1)/nth;
    for (int64_t row = begin; row < end; ++row) {
        source.data = static_cast<uint8_t *>(projection->data) + row*projection->nb[1];
        prior.data = static_cast<uint8_t *>(residual->data) + row*residual->nb[1];
        biased.data = result.data = static_cast<uint8_t *>(dst->data) + row*dst->nb[1];
        ggml_compute_forward_add(&single, &biased);
        ggml_compute_forward_add(&single, &result);
    }
}

bool positive_zero_bias(const float * data, size_t bytes) {
    for (size_t i = 0; i < bytes / sizeof(float); ++i) {
        uint32_t bits; std::memcpy(&bits, data + i, sizeof(bits));
        if (bits != 0) return false;
    }
    return true;
}

// A physical CUSTOM result owns bytes, not logical mixed-dtype model state.
// GGML's public tensor metadata supports views but its view constructors retain
// the source dtype. Construct a bounded byte view first, then describe the typed
// plane with the SAME byte extent. Gallocr still sees the real CUSTOM producer
// through both view_src and src[0]; no pointer or output lives outside its arena.
ggml_tensor * typed_plane(ggml_context * ctx, ggml_tensor * storage, ggml_type type,
                          int64_t width, int64_t rows, size_t offset) {
    require(storage->type == GGML_TYPE_I8 && ggml_is_contiguous(storage) &&
            (type == GGML_TYPE_F32 || type == GGML_TYPE_F16) && width > 0 && rows > 0,
            "invalid typed attention plane");
    const size_t element = ggml_type_size(type);
    require(uint64_t(width) <= SIZE_MAX / element &&
            uint64_t(rows) <= SIZE_MAX / (size_t(width) * element), "attention plane extent overflow");
    const size_t row_bytes = size_t(width) * element, bytes = size_t(rows) * row_bytes;
    require(offset % element == 0 && offset <= ggml_nbytes(storage) &&
            bytes <= ggml_nbytes(storage) - offset, "attention plane exceeds storage");
    auto * view = ggml_view_1d(ctx, storage, int64_t(bytes), offset);
    view->type = type;
    view->ne[0] = width; view->ne[1] = rows; view->ne[2] = view->ne[3] = 1;
    view->nb[0] = element; view->nb[1] = row_bytes;
    view->nb[2] = view->nb[3] = bytes;
    require(view->view_src == storage && view->src[0] == storage && view->view_offs == offset &&
            ggml_nbytes(view) == bytes && ggml_is_contiguous(view), "attention plane alias differs");
    return view;
}

template<bool SkipKeyBias>
void qkv_scatter(ggml_tensor * dst, int ith, int nth, void *) {
    const auto * source = dst->src[0];
    const auto * qb = dst->src[1]; const auto * kb = dst->src[2]; const auto * vb = dst->src[3];
    const int64_t width = qb->ne[0], rows = source->ne[1], padded = GGML_PAD(rows, 256);
    GGML_ASSERT(width == 384 && source->type == GGML_TYPE_F32 && source->ne[0] == 3*width);
    GGML_ASSERT(dst->type == GGML_TYPE_I8 && nth > 0 && ith >= 0 && ith < nth);
    const size_t q_bytes = size_t(width) * rows * sizeof(float);
    const size_t kv_bytes = size_t(width) * padded * sizeof(ggml_fp16_t);
    GGML_ASSERT(ggml_nbytes(dst) == q_bytes + 2*kv_bytes);
    auto * q = reinterpret_cast<float *>(dst->data);
    auto * k = reinterpret_cast<ggml_fp16_t *>(static_cast<uint8_t *>(dst->data) + q_bytes);
    auto * v = reinterpret_cast<ggml_fp16_t *>(static_cast<uint8_t *>(dst->data) + q_bytes + kv_bytes);
    const auto * q_bias = static_cast<const float *>(qb->data);
    const auto * k_bias = static_cast<const float *>(kb->data);
    const auto * v_bias = static_cast<const float *>(vb->data);
    float key[384], value[384];
    for (int64_t row = rows * ith / nth; row < rows * (ith + 1) / nth; ++row) {
        const auto * input = reinterpret_cast<const float *>(static_cast<const uint8_t *>(source->data) + row*source->nb[1]);
        for (int64_t i = 0; i < width; ++i) {
            q[row*width+i] = input[i] + q_bias[i];
            if constexpr (!SkipKeyBias) key[i] = input[width+i] + k_bias[i];
            value[i] = input[2*width+i] + v_bias[i];
        }
        // Same conversion primitive/rounding as the original F32 -> F16 CPY.
        ggml_fp32_to_fp16_row(SkipKeyBias ? input+width : key, k+row*width, width);
        ggml_fp32_to_fp16_row(value, v+row*width, width);
    }
    for (int64_t row = rows + ith; row < padded; row += nth) {
        std::memset(k+row*width, 0, size_t(width)*sizeof(ggml_fp16_t));
        std::memset(v+row*width, 0, size_t(width)*sizeof(ggml_fp16_t));
    }
}

struct Encoder {
    cpu_encoder_config c;
    bool stock_q8;
    bool fused_qkv = false;
    bool tiled_ffn = false;
    std::array<ggml_tensor *, 2> conv_q8 = {};
    ggml_backend_t backend = nullptr;
    ggml_threadpool_t pool = nullptr;
    ggml_context * weights_ctx = nullptr;
    ggml_context * dense_ctx = nullptr;
    ggml_context * graph_ctx = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    ggml_backend_buffer_t dense_buffer = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * padded_k = nullptr;
    ggml_tensor * padded_v = nullptr;
    std::vector<bool> zero_key_bias;
    ggml_cplan plan = {};
    std::vector<uint8_t> scratch;
    std::map<std::string, ggml_tensor *> weights;
    std::map<std::string, std::array<const cpu_encoder_weight *, 3>> qkv_sources;
    std::atomic_flag busy = ATOMIC_FLAG_INIT;

    explicit Encoder(cpu_encoder_config config, bool q8 = false): c(config), stock_q8(q8) {}
    ~Encoder() {
        // Compute is synchronous and the program owner has drained its slot.
        if (backend) ggml_backend_synchronize(backend);
        if (allocator) ggml_gallocr_free(allocator);
        if (graph_ctx) ggml_free(graph_ctx);
        if (weights_buffer) ggml_backend_buffer_free(weights_buffer);
        if (dense_buffer) ggml_backend_buffer_free(dense_buffer);
        if (weights_ctx) ggml_free(weights_ctx);
        if (dense_ctx) ggml_free(dense_ctx);
        if (backend) ggml_backend_free(backend);
        if (pool) ggml_threadpool_free(pool);
    }

    ggml_tensor * w(const std::string & name) { return weights.at(name); }
    ggml_tensor * linear(const std::string & prefix, ggml_tensor * x) {
        return ggml_add(graph_ctx, ggml_mul_mat(graph_ctx, w(prefix + ".weight"), x), w(prefix + ".bias"));
    }
    ggml_tensor * linear_residual(const std::string & prefix, ggml_tensor * x,
                                  ggml_tensor * residual) {
        if (!stock_q8) return ggml_add(graph_ctx, residual, linear(prefix, x));
        auto * projected = ggml_mul_mat(graph_ctx, w(prefix + ".weight"), x);
        auto * joined = ggml_map_custom3_inplace(graph_ctx, projected, w(prefix + ".bias"), residual,
                                               encoder_bias_residual, GGML_N_TASKS_MAX, nullptr);
        ggml_set_name(joined, "encoder_bias_residual");
        return joined;
    }
    ggml_tensor * norm(const std::string & prefix, ggml_tensor * x) {
        if (stock_q8) {
            auto * result = ggml_map_custom3(graph_ctx, x, w(prefix + ".weight"), w(prefix + ".bias"),
                encoder_norm_affine, GGML_N_TASKS_MAX, nullptr);
            ggml_set_name(result, "encoder_norm_affine");
            return result;
        }
        return ggml_add(graph_ctx, ggml_mul(graph_ctx, ggml_norm(graph_ctx, x, 1.e-5f),
                                          w(prefix + ".weight")), w(prefix + ".bias"));
    }
    ggml_tensor * ffn_tiled(const std::string & prefix, ggml_tensor * residual) {
        require(tiled_ffn && stock_q8 && residual->type == GGML_TYPE_F32 &&
                residual->ne[0] == c.width && residual->ne[1] == c.positions &&
                residual->ne[2] == 1 && residual->ne[3] == 1 &&
                residual->nb[0] == sizeof(float), "invalid tiled FFN input");
        // This root is graph-owned output storage, not a semantic state/cache.
        // No read observes it until every contiguous row block has been set.
        auto * assembled = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, c.width, c.positions);
        const size_t stride = assembled->nb[1];
        for (int first=0; first<c.positions; first+=128) {
            const int rows = std::min(128, c.positions-first);
            auto * prior = ggml_view_2d(graph_ctx, residual, c.width, rows,
                                       residual->nb[1], size_t(first)*residual->nb[1]);
            auto * normalized = norm(prefix + ".final_layer_norm", prior);
            auto * projected = ggml_mul_mat(graph_ctx, w(prefix + ".fc1.weight"), normalized);
            auto * expanded = ggml_map_custom2_inplace(graph_ctx, projected, w(prefix + ".fc1.bias"),
                encoder_bias_gelu, GGML_N_TASKS_MAX, nullptr);
            ggml_set_name(expanded, "encoder_ffn_bias_gelu");
            auto * result = linear_residual(prefix + ".fc2", expanded, prior);
            // SET returns an alias of its actual destination and retains both
            // predecessor and result as sources. The ordered chain keeps each
            // complete block live through its copy, then allows arena reuse.
            assembled = ggml_set_2d_inplace(graph_ctx, assembled, result, stride,
                                           size_t(first)*stride);
            ggml_set_name(assembled, "encoder_ffn_tile_join");
        }
        return assembled;
    }
    ggml_tensor * activation(ggml_tensor * x) {
        return stock_q8 ? ggml_gelu(graph_ctx, x) : ggml_gelu_erf(graph_ctx, x);
    }
    ggml_tensor * attention(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v) {
        const int d = c.width / c.heads;
        // Preserve HF's Q scaling before attention (not a changed softmax scale).
        if (!stock_q8) q = ggml_scale(graph_ctx, q, 1.f / std::sqrt(float(d)));
        // Match public whisper.cpp's Q8 encoder equation: Q has the full
        // logical extent, while unmasked F16 K/V includes zero-padded rows up
        // to the flash kernel's 256-position boundary. Zero keys participate
        // in the denominator; this is an explicit physical-math difference
        // from the former 1500-key experiment, not an output crop or mask.
        const int kv_positions = stock_q8 ? GGML_PAD(c.positions, 256) : c.positions;
        if (stock_q8) {
            // Same ordered prefix copies as whisper_build_graph_encoder in
            // ggml-org/whisper.cpp (MIT, pinned by build.py). The arena roots
            // have immutable zero tails and are reused only by this synchronous
            // encoder slot; every live prefix is overwritten before each use.
            ggml_build_forward_expand(graph, ggml_cpy(graph_ctx, k,
                ggml_view_1d(graph_ctx, padded_k, int64_t(c.width) * c.positions, 0)));
            ggml_build_forward_expand(graph, ggml_cpy(graph_ctx, v,
                ggml_view_1d(graph_ctx, padded_v, int64_t(c.width) * c.positions, 0)));
            k = ggml_reshape_2d(graph_ctx, padded_k, c.width, kv_positions);
            v = ggml_reshape_2d(graph_ctx, padded_v, c.width, kv_positions);
        }
        auto split = [&](ggml_tensor * x, int positions) {
            return ggml_permute(graph_ctx, ggml_reshape_3d(graph_ctx, x, d, c.heads, positions), 0, 2, 1, 3);
        };
        q = split(q, c.positions); k = split(k, kv_positions); v = split(v, kv_positions);
        // The F32 entry retains its original unpadded strided F32 K/V.
        auto * attended = ggml_flash_attn_ext(graph_ctx, q, k, v, nullptr,
                                             stock_q8 ? 1.f / std::sqrt(float(d)) : 1.f, 0.f, 0.f);
#if defined(MSTAR_ENCODER_QK_SDOT)
        if (stock_q8) { const int32_t tag=MSTAR_ENCODER_QK_SDOT_TAG;
            std::memcpy((char *)attended->op_params+4*sizeof(tag), &tag, sizeof(tag)); }
#endif
#if defined(MSTAR_ENCODER_PV_SDOT)
        if (stock_q8) { const int32_t tag=MSTAR_ENCODER_PV_SDOT_TAG;
            std::memcpy((char *)attended->op_params+5*sizeof(tag), &tag, sizeof(tag)); }
#endif
        if (!stock_q8) ggml_prec_set_acc(attended, GGML_PREC_F32);
        return ggml_reshape_2d(graph_ctx, attended, c.width, c.positions);
    }

    ggml_tensor * fused_attention(ggml_tensor * all, const std::string & prefix, bool skip_key_bias) {
        const int padded = GGML_PAD(c.positions, 256), d = c.width/c.heads;
        const size_t q_bytes = size_t(c.width)*c.positions*sizeof(float);
        const size_t kv_bytes = size_t(c.width)*padded*sizeof(ggml_fp16_t);
        ggml_tensor * args[] = {all, w(prefix+".q_proj.bias"), w(prefix+".k_proj.bias"), w(prefix+".v_proj.bias")};
        auto * storage = ggml_custom_4d(graph_ctx, GGML_TYPE_I8, q_bytes+2*kv_bytes, 1, 1, 1,
            args, 4, skip_key_bias ? qkv_scatter<true> : qkv_scatter<false>, GGML_N_TASKS_MAX, nullptr);
        ggml_set_name(storage, "encoder_qkv_bias_scatter_cast");
        auto * q = typed_plane(graph_ctx, storage, GGML_TYPE_F32, c.width, c.positions, 0);
        auto * k = typed_plane(graph_ctx, storage, GGML_TYPE_F16, c.width, padded, q_bytes);
        auto * v = typed_plane(graph_ctx, storage, GGML_TYPE_F16, c.width, padded, q_bytes+kv_bytes);
        auto split = [&](ggml_tensor * x, int rows) {
            return ggml_permute(graph_ctx, ggml_reshape_3d(graph_ctx, x, d, c.heads, rows), 0, 2, 1, 3);
        };
        // All operands retain the same CUSTOM producer as their view_src. The
        // existing node completion barrier publishes every plane before FLASH.
        auto * attended = ggml_flash_attn_ext(graph_ctx, split(q,c.positions), split(k,padded),
            split(v,padded), nullptr, 1.f/std::sqrt(float(d)), 0.f, 0.f);
#if defined(MSTAR_ENCODER_QK_SDOT)
        const int32_t tag=MSTAR_ENCODER_QK_SDOT_TAG;
        std::memcpy((char *)attended->op_params+4*sizeof(tag), &tag, sizeof(tag));
#endif
#if defined(MSTAR_ENCODER_PV_SDOT)
        { const int32_t tag=MSTAR_ENCODER_PV_SDOT_TAG;
          std::memcpy((char *)attended->op_params+5*sizeof(tag), &tag, sizeof(tag)); }
#endif
        return ggml_reshape_2d(graph_ctx, attended, c.width, c.positions);
    }

    void prepare(const cpu_encoder_weight * descriptors, size_t count, cpu_encoder_memory * memory) {
        // Match the unchanged declaration's supported logical configuration.
        require(c.width == 384 && c.layers == 4 && c.heads == 6 && c.ffn == 1536 &&
                c.mels == 80 && c.positions == 1500 && c.threads >= 1 && c.threads <= 64,
                "unsupported CPU encoder configuration");
        require(descriptors && count > 0 && count < 256, "invalid weight descriptors");
        backend = ggml_backend_cpu_init();
        require(backend, "CPU backend initialization failed");
        auto params = ggml_threadpool_params_default(c.threads);
        // Sleep at idle rather than stealing the decoder's cores between calls.
        params.poll = 0;
        pool = ggml_threadpool_new(&params);
        require(pool, "CPU threadpool initialization failed");
        ggml_backend_cpu_set_n_threads(backend, c.threads);
        ggml_backend_cpu_set_threadpool(backend, pool);

        const size_t weights_metadata = (count + 1) * ggml_tensor_overhead();
        weights_ctx = ggml_init({weights_metadata, nullptr, true});
        require(weights_ctx, "weight metadata allocation failed");
        if (stock_q8) {
            dense_ctx = ggml_init({weights_metadata, nullptr, true});
            require(dense_ctx, "quantized weight metadata allocation failed");
        }
        std::map<std::string, const cpu_encoder_weight *> source;
        for (size_t i = 0; i < count; ++i) {
            const auto & d = descriptors[i];
            require(d.name && std::strlen(d.name) < 128 && aligned(d.data), "invalid weight name or pointer");
            require(d.rank >= 1 && d.rank <= 3, "invalid weight rank");
            uint64_t bytes = sizeof(float);
            for (int axis = 0; axis < d.rank; ++axis) {
                require(d.ne[axis] > 0 && d.ne[axis] <= 100000 && bytes <= UINT64_MAX / uint64_t(d.ne[axis]),
                        "invalid weight extent");
                bytes *= uint64_t(d.ne[axis]);
            }
            require(bytes == d.bytes && source.emplace(d.name, &d).second, "duplicate or non-F32 weight extent");
        }
        std::set<std::string> consumed;
        auto claim = [&](const std::string & name, std::initializer_list<int64_t> shape) {
            auto it = source.find(name);
            require(it != source.end(), "missing model-owned encoder weight");
            const auto & d = *it->second;
            require(size_t(d.rank) == shape.size(), "weight rank differs from configuration");
            int axis = 0;
            for (auto ne : shape) require(d.ne[axis++] == ne, "weight shape differs from configuration");
            require(consumed.insert(name).second, "encoder descriptor used twice");
            return &d;
        };
        auto add = [&](const std::string & name, std::initializer_list<int64_t> shape) {
            const auto & d = *claim(name, shape);
            const bool dense = stock_q8 && name.find("encoder.layers.") == 0 && d.rank == 2;
            const bool conv = stock_q8 && name.find("encoder.conv") == 0 && d.rank == 3;
            auto * t = ggml_new_tensor(dense ? dense_ctx : weights_ctx,
                                      dense ? GGML_TYPE_Q8_0 : conv ? GGML_TYPE_F16 : GGML_TYPE_F32, d.rank, d.ne);
            ggml_set_name(t, name.c_str());
            weights.emplace(name, t);
        };
        auto add_linear = [&](const std::string & p, int in, int out) {
            add(p + ".weight", {in, out}); add(p + ".bias", {out});
        };
        auto add_norm = [&](const std::string & p) { add(p + ".weight", {c.width}); add(p + ".bias", {c.width}); };
        add("encoder.conv1.weight", {3, c.mels, c.width}); add("encoder.conv1.bias", {c.width});
        add("encoder.conv2.weight", {3, c.width, c.width}); add("encoder.conv2.bias", {c.width});
        add("encoder.embed_positions.weight", {c.width, c.positions});
        add_norm("encoder.layer_norm");
        for (int layer = 0; layer < c.layers; ++layer) {
            const std::string p = "encoder.layers." + std::to_string(layer);
            add_norm(p + ".self_attn_layer_norm"); add_norm(p + ".final_layer_norm");
            if (stock_q8) {
                const std::string name = p + ".self_attn.qkv.weight";
                std::array<const cpu_encoder_weight *, 3> parts;
                int i = 0;
                for (const char * projection : {"q_proj", "k_proj", "v_proj"}) {
                    const auto prefix = p + ".self_attn." + projection;
                    parts[i++] = claim(prefix + ".weight", {c.width, c.width});
                    add(prefix + ".bias", {c.width});
                }
                auto * tensor = ggml_new_tensor_2d(dense_ctx, GGML_TYPE_Q8_0, c.width, 3*c.width);
                ggml_set_name(tensor, name.c_str()); weights.emplace(name, tensor);
                qkv_sources.emplace(name, parts);
                fused_qkv = true;
            } else {
                for (const char * projection : {"q_proj", "k_proj", "v_proj"})
                    add_linear(p + ".self_attn." + projection, c.width, c.width);
            }
            add_linear(p + ".self_attn.out_proj", c.width, c.width);
            add_linear(p + ".fc1", c.width, c.ffn); add_linear(p + ".fc2", c.ffn, c.width);
        }
        require(consumed.size() == source.size(), "unexpected encoder weight");
        weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, backend);
        require(weights_buffer, "weight buffer allocation failed");
        ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (stock_q8) {
            // Same ordered public extra-buft selection as whisper.cpp. Each
            // admitted provider must support every Q8 matrix/query geometry.
            auto * device = ggml_backend_get_device(backend);
            auto * registry = ggml_backend_dev_backend_reg(device);
            auto get_extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
                ggml_backend_reg_get_proc_address(registry, "ggml_backend_dev_get_extra_bufts"));
            auto * choices = get_extra ? get_extra(device) : nullptr;
            auto supports = [&](ggml_backend_buffer_type_t candidate, const std::string & name,
                                int64_t inside, int64_t outside, int queries = 0) {
                if (!queries) queries = c.positions;
                auto * probe = ggml_init({3 * ggml_tensor_overhead(), nullptr, true});
                require(probe, "CPU Q8 provider probe metadata allocation failed");
                auto * weight = ggml_new_tensor_2d(probe, GGML_TYPE_Q8_0, inside, outside);
                auto * input = ggml_new_tensor_2d(probe, GGML_TYPE_F32, inside, queries);
                auto * op = ggml_mul_mat(probe, weight, input);
                weight->buffer = ggml_backend_buft_alloc_buffer(candidate, 0);
                const bool supported = weight->buffer && ggml_backend_dev_supports_op(device, op);
                if (weight->buffer) ggml_backend_buffer_free(weight->buffer);
                weight->buffer = nullptr; ggml_free(probe);
                if (!supported)
                    std::fprintf(stderr, "CPU_GGML_Q8_PROVIDER_REJECT role=encoder provider=%s weight=%s queries=%d rows=%lld\n",
                                 ggml_backend_buft_name(candidate), name.c_str(), queries, (long long)outside);
                return supported;
            };
            // Choose the SAME ordered provider as the original three matrices.
            // A fused-geometry refusal falls back to that original physical
            // topology, not to another provider/quantization or a smaller domain.
            auto selected = mstar_cpu_ggml::first_supported_q8_provider(
                choices, ggml_backend_buft_name, [&](ggml_backend_buffer_type_t candidate) {
                    for (const auto & entry : source) {
                        const auto & d = *entry.second;
                        if (entry.first.find("encoder.layers.") != 0 || d.rank != 2) continue;
                        if (!supports(candidate, entry.first, d.ne[0], d.ne[1])) return false;
                    }
                    return true;
                });
            require(selected, "actual CPU_KLEIDIAI/CPU_REPACK Q8 matrix support is unavailable");
            for (const auto & entry : qkv_sources)
                if (!supports(selected, entry.first, c.width, 3*c.width)) fused_qkv = false;
            if (!fused_qkv) {
                for (auto it = weights.begin(); it != weights.end();)
                    if (it->second->type == GGML_TYPE_Q8_0) it = weights.erase(it); else ++it;
                ggml_free(dense_ctx);
                dense_ctx = ggml_init({weights_metadata, nullptr, true});
                require(dense_ctx, "unfused Q8 metadata allocation failed");
                for (const auto & entry : source) {
                    const auto & d = *entry.second;
                    if (entry.first.find("encoder.layers.") != 0 || d.rank != 2) continue;
                    auto * tensor = ggml_new_tensor_2d(dense_ctx, GGML_TYPE_Q8_0, d.ne[0], d.ne[1]);
                    ggml_set_name(tensor, entry.first.c_str()); weights.emplace(entry.first, tensor);
                }
                qkv_sources.clear();
            }
            std::fprintf(stderr, "CPU_GGML_ENCODER_QKV mode=%s reason=%s provider=%s rows=%d\n",
                         fused_qkv ? "fused" : "separate", fused_qkv ? "supported" : "fused_geometry_refused",
                         ggml_backend_buft_name(selected), fused_qkv ? 3*c.width : c.width);
            for (int i = 0; i < 2; ++i) {
                if (!(MSTAR_ENCODER_Q8_CONV & (1 << i))) continue;
                const std::string name = "encoder.conv" + std::to_string(i + 1) + ".weight";
                const int inside = GGML_PAD(3 * (i ? c.width : c.mels), 32);
                if (supports(selected, name + ".q8", inside, c.width, (i ? 1 : 2) * c.positions)) {
                    conv_q8[i] = ggml_new_tensor_2d(dense_ctx, GGML_TYPE_Q8_0, inside, c.width);
                    ggml_set_name(conv_q8[i], (name + ".q8").c_str());
                }
                std::fprintf(stderr, "CPU_GGML_ENCODER_CONV layer=%d mode=%s provider=%s padded_k=%d reason=%s\n",
                    i + 1, conv_q8[i] ? "q8" : "f16", ggml_backend_buft_name(selected), inside,
                    conv_q8[i] ? "supported" : "derived_geometry_refused");
            }
            if (MSTAR_ENCODER_FFN_TILE > 0 && c.positions > MSTAR_ENCODER_FFN_TILE) {
                tiled_ffn = true;
                const int tail = c.positions % 128;
                for (const auto & entry : source) {
                    const auto & d = *entry.second;
                    if (entry.first.find("encoder.layers.") != 0 || d.rank != 2 ||
                        (entry.first.find(".fc1.weight") == std::string::npos &&
                         entry.first.find(".fc2.weight") == std::string::npos)) continue;
                    if (!supports(selected, entry.first, d.ne[0], d.ne[1], 128) ||
                        (tail && !supports(selected, entry.first, d.ne[0], d.ne[1], tail))) tiled_ffn = false;
                }
            }
            std::fprintf(stderr, "CPU_GGML_ENCODER_FFN mode=%s tile=%d provider=%s reason=%s\n",
                tiled_ffn ? "query_tiled" : "full_query", MSTAR_ENCODER_FFN_TILE,
                ggml_backend_buft_name(selected), tiled_ffn ? "supported" :
                (MSTAR_ENCODER_FFN_TILE > 0 ? "geometry_refused_or_not_smaller" : "default"));
            dense_buffer = ggml_backend_alloc_ctx_tensors_from_buft(dense_ctx, selected);
            require(dense_buffer, "Q8 provider weight allocation failed");
            ggml_backend_buffer_set_usage(dense_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            std::fprintf(stderr, "CPU_GGML_Q8_PROVIDER role=encoder provider=%s buffer=%s order=upstream_extra_bufts\n",
                         ggml_backend_buft_name(selected), ggml_backend_buffer_name(dense_buffer));
        }
        for (const auto & entry : weights) {
            auto * tensor = entry.second;
            const auto group = qkv_sources.find(entry.first);
            std::vector<float> joined(group == qkv_sources.end() ? 0 : size_t(3) * c.width * c.width);
            const float * values;
            size_t bytes;
            if (group == qkv_sources.end()) {
                const auto & d = *source.at(entry.first); values = d.data; bytes = d.bytes;
            } else {
                size_t offset = 0;
                for (const auto * part : group->second) {
                    std::memcpy(joined.data() + offset, part->data, part->bytes);
                    offset += part->bytes / sizeof(float);
                }
                require(offset == joined.size(), "fused QKV descriptor extent differs");
                values = joined.data(); bytes = joined.size() * sizeof(float);
            }
            if (tensor->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(tensor, values, 0, bytes);
            } else {
                const size_t n = bytes / sizeof(float);
                std::vector<ggml_fp16_t> half(n);
                ggml_cpu_fp32_to_fp16(values, half.data(), n);
                if (tensor->type == GGML_TYPE_F16) {
                    ggml_backend_tensor_set(tensor, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
                } else {
                    // Stock Q8 model conversion quantizes the F16 checkpoint.
                    std::vector<float> rounded(n);
                    ggml_cpu_fp16_to_fp32(half.data(), rounded.data(), n);
                    std::vector<uint8_t> canonical(ggml_nbytes(tensor));
                    require(ggml_quantize_chunk(GGML_TYPE_Q8_0, rounded.data(), canonical.data(), 0,
                                                tensor->ne[1], tensor->ne[0], nullptr) == canonical.size(), "Q8 packing size differs");
                    // Whole canonical tensor upload invokes the upstream provider packer.
                    ggml_backend_tensor_set(tensor, canonical.data(), 0, canonical.size());
                }
            }
        }

        // Derive each immutable matrix from the same F16-rounded checkpoint as
        // the original convolution; only added K columns are zero. Whole upload
        // invokes the selected provider's existing Q8_0 repacker.
        for (int i = 0; i < 2; ++i) if (auto * tensor = conv_q8[i]) {
            const auto & d = *source.at("encoder.conv" + std::to_string(i + 1) + ".weight");
            const int inside = 3 * (i ? c.width : c.mels);
            std::vector<ggml_fp16_t> half(d.bytes / sizeof(float));
            ggml_cpu_fp32_to_fp16(d.data, half.data(), half.size());
            std::vector<float> rounded(size_t(tensor->ne[0]) * c.width, 0.f);
            for (int row = 0; row < c.width; ++row)
                ggml_cpu_fp16_to_fp32(half.data() + row*inside, rounded.data() + row*tensor->ne[0], inside);
            std::vector<uint8_t> canonical(ggml_nbytes(tensor));
            require(ggml_quantize_chunk(GGML_TYPE_Q8_0, rounded.data(), canonical.data(), 0,
                c.width, tensor->ne[0], nullptr) == canonical.size(), "convolution Q8 packing size differs");
            ggml_backend_tensor_set(tensor, canonical.data(), 0, canonical.size());
        }

        qkv_sources.clear();
        const size_t graph_metadata = max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false);
        graph_ctx = ggml_init({graph_metadata, nullptr, true});
        require(graph_ctx, "graph metadata allocation failed");
        graph = ggml_new_graph_custom(graph_ctx, max_nodes, false);
        input = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, 2 * c.positions, c.mels);
        ggml_set_name(input, "audio_features"); ggml_set_input(input);
        if (stock_q8 && !fused_qkv) {
            const int64_t elements = int64_t(c.width) * GGML_PAD(c.positions, 256);
            padded_k = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_F16, elements);
            padded_v = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_F16, elements);
            ggml_set_name(padded_k, "encoder_attention_padded_k");
            ggml_set_name(padded_v, "encoder_attention_padded_v");
            // Output flags reserve these arena roots across the full graph:
            // gallocr must not recycle their zero tails after the last layer.
            ggml_set_input(padded_k); ggml_set_output(padded_k);
            ggml_set_input(padded_v); ggml_set_output(padded_v);
        }
        if (stock_q8) {
            zero_key_bias.resize(c.layers, true);
            for (int layer = 0; layer < c.layers; ++layer) {
                const auto & bias = *source.at("encoder.layers." + std::to_string(layer) + ".self_attn.k_proj.bias");
                // Public Whisper has no key bias. Elide only a supplied +0
                // vector; nonzero, nonfinite and signed-zero values retain ADD.
                zero_key_bias[layer] = positive_zero_bias(bias.data, bias.bytes);
            }
        }
        auto * x = input;
        ggml_tensor * y = nullptr;
        for (int i = 0; i < 2; ++i) {
            const std::string p = "encoder.conv" + std::to_string(i + 1);
            const int channels = i ? c.width : c.mels;
            auto * kernel = w(p + ".weight");
            if (stock_q8) {
                if (conv_q8[i]) {
                    y = activation(ggml_add(graph_ctx, convolution_q8(graph_ctx, kernel, conv_q8[i], x, i ? 2 : 1), w(p + ".bias")));
                    // KAI gives [channel,time]. The next upstream IM2COL needs
                    // contiguous time; the final encoder already wants this layout.
                    if (!i) x = ggml_cont(graph_ctx, ggml_transpose(graph_ctx, y));
                    continue;
                }
                y = ggml_conv_1d_ph(graph_ctx, kernel, x, i ? 2 : 1, 1);
                y = activation(ggml_add(graph_ctx, y, ggml_reshape_2d(graph_ctx, w(p + ".bias"), 1, c.width)));
                x = y;
                continue;
            }
            auto * columns = ggml_im2col(graph_ctx, kernel, x, i ? 2 : 1, 0, 1, 0, 1, 0, false, GGML_TYPE_F32);
            y = ggml_mul_mat(graph_ctx, ggml_reshape_2d(graph_ctx, kernel, 3 * channels, c.width), columns);
            y = ggml_gelu_erf(graph_ctx, ggml_add(graph_ctx, y, w(p + ".bias")));
            if (!i) x = ggml_cont(graph_ctx, ggml_transpose(graph_ctx, y));
        }
        if (stock_q8 && !conv_q8[1]) y = ggml_cont(graph_ctx, ggml_transpose(graph_ctx, y));
        x = ggml_add(graph_ctx, y, w("encoder.embed_positions.weight"));
        for (int layer = 0; layer < c.layers; ++layer) {
            const std::string p = "encoder.layers." + std::to_string(layer);
            auto * n = norm(p + ".self_attn_layer_norm", x);
            ggml_tensor * a;
            if (stock_q8 && fused_qkv) {
                auto * all = ggml_mul_mat(graph_ctx, w(p + ".self_attn.qkv.weight"), n);
                a = fused_attention(all, p+".self_attn", zero_key_bias[layer]);
            } else if (stock_q8) {
                auto * key = zero_key_bias[layer]
                    ? ggml_mul_mat(graph_ctx, w(p + ".self_attn.k_proj.weight"), n)
                    : linear(p + ".self_attn.k_proj", n);
                a = attention(linear(p + ".self_attn.q_proj", n), key, linear(p + ".self_attn.v_proj", n));
            } else {
                a = attention(linear(p + ".self_attn.q_proj", n), linear(p + ".self_attn.k_proj", n),
                              linear(p + ".self_attn.v_proj", n));
            }
            x = linear_residual(p + ".self_attn.out_proj", a, x);
            if (tiled_ffn) {
                x = ffn_tiled(p, x);
                continue;
            }
            n = norm(p + ".final_layer_norm", x);
            ggml_tensor * expanded;
            if (stock_q8) {
                auto * projected = ggml_mul_mat(graph_ctx, w(p + ".fc1.weight"), n);
                // This internal matrix has only this consumer; retain its arena
                // storage just as the original ADD/UNARY allocator reuse did.
                expanded = ggml_map_custom2_inplace(graph_ctx, projected, w(p + ".fc1.bias"),
                    encoder_bias_gelu, GGML_N_TASKS_MAX, nullptr);
                ggml_set_name(expanded, "encoder_ffn_bias_gelu");
            } else {
                expanded = activation(linear(p + ".fc1", n));
            }
            x = linear_residual(p + ".fc2", expanded, x);
        }
        output = norm("encoder.layer_norm", x);
        ggml_set_name(output, "encoder_states"); ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        require(ggml_graph_n_nodes(graph) <= int(max_nodes), "encoder graph node budget exceeded");
        // No scheduler that could substitute another backend or convert types.
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (!stock_q8 && node->op == GGML_OP_MUL_MAT) {
                require(ggml_prec_set_acc(node, GGML_PREC_F32) && ggml_prec_set_src(node, GGML_PREC_F32, 1),
                        "CPU matrix precision could not be fixed to F32");
            }
            require((stock_q8 || node->type == GGML_TYPE_F32) && ggml_backend_supports_op(backend, node),
                    "CPU encoder graph operation is unsupported");
        }
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(allocator && ggml_gallocr_alloc_graph(allocator, graph), "graph workspace allocation failed");
        if (stock_q8 && !fused_qkv) {
            const std::vector<ggml_fp16_t> zeros(ggml_nelements(padded_k), ggml_fp32_to_fp16(0.f));
            ggml_backend_tensor_set(padded_k, zeros.data(), 0, ggml_nbytes(padded_k));
            ggml_backend_tensor_set(padded_v, zeros.data(), 0, ggml_nbytes(padded_v));
        }
        require(ggml_is_contiguous(output) && ggml_nbytes(output) == uint64_t(c.width) * c.positions * sizeof(float),
                "encoder output layout differs");
        // Retain the upstream CPU plan/scratch once, not a per-call allocation.
        plan = ggml_graph_plan(graph, c.threads, pool);
        scratch.resize(plan.work_size);
        plan.work_data = scratch.data();
        if (memory) *memory = {ggml_backend_buffer_get_size(weights_buffer) + (dense_buffer ? ggml_backend_buffer_get_size(dense_buffer) : 0),
                              ggml_gallocr_get_buffer_size(allocator, 0), weights_metadata * (stock_q8 ? 2 : 1) + graph_metadata, plan.work_size};
    }
};
struct BusyGuard {
    std::atomic_flag & flag;
    explicit BusyGuard(std::atomic_flag & f): flag(f) {
        require(!flag.test_and_set(std::memory_order_acquire), "CPU encoder workspace is busy");
    }
    ~BusyGuard() { flag.clear(std::memory_order_release); }
};
} // namespace

extern "C" {
int cpu_encoder_abi(void) { return 1; }
int cpu_encoder_q8_abi(void) { return 1; }
int cpu_encoder_q8_capability(void) {
    // Same public supports_op probe as upstream's weight-buft selection. No
    // model execution or model/request storage is created by discovery.
    auto * backend = ggml_backend_cpu_init();
    if (!backend) return 0;
    auto * device = ggml_backend_get_device(backend);
    auto get_extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(device), "ggml_backend_dev_get_extra_bufts"));
    auto * ctx = ggml_init({3 * ggml_tensor_overhead(), nullptr, true});
    if (!ctx) { ggml_backend_free(backend); return 0; }
    auto * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 32, 4);
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 1500);
    auto * op = ggml_mul_mat(ctx, w, x);
    bool supported = false;
    for (auto * b = get_extra ? get_extra(device) : nullptr; b && *b; ++b) {
        if (std::strcmp(ggml_backend_buft_name(*b), "CPU_REPACK") != 0) continue;
        w->buffer = ggml_backend_buft_alloc_buffer(*b, 0);
        supported = w->buffer && ggml_backend_dev_supports_op(device, op);
        if (w->buffer) ggml_backend_buffer_free(w->buffer);
        w->buffer = nullptr;
        if (supported) break;
    }
    ggml_free(ctx); ggml_backend_free(backend);
    return supported ? 1 : 0;
}
void * cpu_encoder_create(const cpu_encoder_config * c, const cpu_encoder_weight * w, size_t n,
                         cpu_encoder_memory * memory, char * error, size_t size) {
    try {
        require(c, "missing encoder configuration");
        auto owner = std::make_unique<Encoder>(*c);
        owner->prepare(w, n, memory);
        return owner.release();
    } catch (const std::exception & e) { error_text(error, size, e.what()); return nullptr; }
}
void * cpu_encoder_q8_create(const cpu_encoder_config * c, const cpu_encoder_weight * w, size_t n,
                            cpu_encoder_memory * memory, char * error, size_t size) {
    try {
        require(c, "missing encoder configuration");
        auto owner = std::make_unique<Encoder>(*c, true);
        owner->prepare(w, n, memory);
        return owner.release();
    } catch (const std::exception & e) { error_text(error, size, e.what()); return nullptr; }
}
int cpu_encoder_run(void * handle, const float * features, uint64_t feature_bytes,
                    float * states, uint64_t state_bytes, char * error, size_t size) {
    try {
        require(handle, "CPU encoder is closed");
        auto & owner = *static_cast<Encoder *>(handle);
        BusyGuard guard(owner.busy);
        require(aligned(features) && aligned(states) && feature_bytes == ggml_nbytes(owner.input) &&
                state_bytes == ggml_nbytes(owner.output) && disjoint(features, feature_bytes, states, state_bytes),
                "invalid or overlapping CPU encoder call storage");
        ggml_backend_tensor_set(owner.input, features, 0, feature_bytes);
        require(ggml_graph_compute(owner.graph, &owner.plan) == GGML_STATUS_SUCCESS, "CPU graph computation failed");
        ggml_backend_tensor_get(owner.output, states, 0, state_bytes);
        return 0;
    } catch (const std::exception & e) { error_text(error, size, e.what()); return -1; }
}
int cpu_encoder_close(void * handle, char * error, size_t size) {
    try {
        if (!handle) return 0;
        auto * owner = static_cast<Encoder *>(handle);
        require(!owner->busy.test_and_set(std::memory_order_acquire), "cannot close a busy CPU encoder");
        delete owner;
        return 0;
    } catch (const std::exception & e) { error_text(error, size, e.what()); return -1; }
}
}
