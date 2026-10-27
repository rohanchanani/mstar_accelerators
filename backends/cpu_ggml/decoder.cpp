// Decoder topology follows ggml-org/whisper.cpp (MIT; see LICENSE.ggml).
// Only physical math/workspace is owned here. All KV is borrowed from M*.
#include "decoder.h"
#include "decoder_storage.h"
#include "decoder_readout.h"
#include "decoder_tokens.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "q8_provider.h"
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t max_nodes = 4096;
void require(bool b, const char * e) { if (!b) throw std::runtime_error(e); }
void message(char * p, size_t n, const char * e) { if (p && n) std::snprintf(p, n, "%s", e); }
struct Range { const void * data; uint64_t bytes; };
void check_range(Range r, uint64_t bytes, size_t alignment) {
    require(r.data && r.bytes == bytes && uintptr_t(r.data) % alignment == 0 &&
            uintptr_t(r.data) <= UINTPTR_MAX - bytes, "invalid borrowed storage extent/alignment");
}
bool disjoint(Range a, Range b) {
    return uintptr_t(a.data) + a.bytes <= uintptr_t(b.data) || uintptr_t(b.data) + b.bytes <= uintptr_t(a.data);
}
void mask_check(const float * mask, int queries, int capacity, int length) {
    for (int row = 0; row < queries; ++row) for (int column = 0; column < capacity; ++column) {
        float x = mask[row * capacity + column];
        require(x == 0.f || x == -INFINITY, "dense attention mask is not exactly zero/-infinity");
        require(column < length || x == -INFINITY, "mask exposes outside admitted valid prefix");
    }
}
void positions_check(const int64_t * p, int n, int capacity, int length) {
    std::array<bool, 1500> seen = {};
    for (int i = 0; i < n; ++i) {
        require(p[i] >= 0 && p[i] < capacity && p[i] < length && !seen[p[i]], "invalid/duplicate admitted write position");
        seen[p[i]] = true;
    }
}
ggml_type storage_type(uint32_t encoding) {
    if (encoding == CPU_DECODER_PLAIN_F32) return GGML_TYPE_F32;
    if (encoding == CPU_DECODER_PLAIN_F16) return GGML_TYPE_F16;
    throw std::runtime_error("unsupported plain KV storage encoding");
}
std::string cross_projection_name(int layer, int part, const char * suffix) {
    return "decoder.layers." + std::to_string(layer) + ".encoder_attn." +
        (part == 0 ? "k_proj." : "v_proj.") + suffix;
}
int cross_projection_index(const std::string & name, int layers) {
    for (int layer = 0; layer < layers; ++layer) for (int part = 0; part < 2; ++part)
        if (name == cross_projection_name(layer, part, "weight")) return 2*layer + part;
    return -1;
}
ggml_tensor * cross_projection_view(ggml_context * ctx, ggml_tensor * joined,
                                    int width, int layers, int source, int layer, int part) {
    require(joined && joined->type == GGML_TYPE_F32 && ggml_is_contiguous(joined) &&
            width > 0 && layers > 0 && source > 0 && layer >= 0 && layer < layers &&
            (part == 0 || part == 1) && joined->ne[0] == 2*layers*width &&
            joined->ne[1] == source && joined->ne[2] == 1 && joined->ne[3] == 1,
            "invalid joined cross-projection view");
    return ggml_view_2d(ctx,joined,width,source,joined->nb[1],size_t(2*layer+part)*width*sizeof(float));
}
// Same F32 ADD followed by the SET_ROWS destination converter, without a
// whole joined biased activation. The public CUSTOM-inplace result aliases
// one admitted owner plane; every source and that write are graph-visible.
void cross_bias_write(ggml_tensor * dst, int ith, int nth, void *) {
    const auto * projected = dst->src[1];
    const auto * bias = dst->src[2];
    const auto * positions = dst->src[3];
    const int width = int(dst->ne[0]);
    GGML_ASSERT(width == 384 && projected->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32 &&
                positions->type == GGML_TYPE_I64 && projected->ne[0] == width &&
                bias->ne[0] == width && positions->ne[0] == projected->ne[1] &&
                projected->nb[0] == sizeof(float) && bias->nb[0] == sizeof(float) &&
                (dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32));
    alignas(64) float values[384];
    const auto * biases = static_cast<const float *>(bias->data);
    for (int64_t row = ith; row < projected->ne[1]; row += nth) {
        const auto index = *reinterpret_cast<const int64_t *>(static_cast<const char *>(positions->data)+row*positions->nb[0]);
        GGML_ASSERT(index >= 0 && index < dst->ne[1]); // complete call validated before any graph write
        const auto * source = reinterpret_cast<const float *>(static_cast<const char *>(projected->data)+row*projected->nb[1]);
        for (int column = 0; column < width; ++column) values[column] = source[column]+biases[column];
        auto * target = static_cast<char *>(dst->data)+index*dst->nb[1];
        if (dst->type == GGML_TYPE_F16) {
            // Exact same compiled function as CPU F16 SET_ROWS.from_float.
            ggml_cpu_fp32_to_fp16(values,reinterpret_cast<ggml_fp16_t *>(target),width);
        } else {
            std::memcpy(target,values,width*sizeof(float));
        }
    }
}
ggml_tensor * cross_bias_set_rows(ggml_context * ctx, ggml_tensor * destination,
        ggml_tensor * projected, ggml_tensor * bias, ggml_tensor * positions) {
    ggml_tensor * args[] = {projected,bias,positions};
    auto * result = ggml_custom_inplace(ctx,destination,args,3,cross_bias_write,GGML_N_TASKS_MAX,nullptr);
    ggml_set_name(result,"decoder.cross_projection.bias_set_rows");
    return result;
}
struct Decoder;
struct Graph {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cplan plan = {}, cross_plan = {};
    ggml_cgraph * body_graph = nullptr, * cross_graph = nullptr;
    std::vector<uint8_t> cross_scratch;
    std::vector<uint8_t> scratch;
    std::vector<ggml_tensor *> roots, borrowed_views;
    ggml_tensor * x = nullptr, * encoder = nullptr, * self_positions = nullptr, * cross_positions = nullptr;
    ggml_tensor * self_mask = nullptr, * cross_mask = nullptr, * output = nullptr;
    ggml_tensor * captured_query = nullptr, * logits = nullptr;
    ggml_tensor * token_ids = nullptr, * position_ids = nullptr;
    bool invalid_readout_input = false;
    std::array<std::array<ggml_tensor *, 4>, 4> kv = {};
    ~Graph() { if (allocator) ggml_gallocr_free(allocator); if (ctx) ggml_free(ctx); }
    void build(Decoder &, const cpu_decoder_storage_call &);
    void bind(const cpu_decoder_storage_call & call) {
        if (x) x->data = const_cast<float *>(call.x);
        self_positions->data = const_cast<int64_t *>(call.self_positions);
        if (encoder) {
            encoder->data = const_cast<float *>(call.encoder);
            cross_positions->data = const_cast<int64_t *>(call.cross_positions);
        }
        for (int i = 0; i < 4; ++i) {
            kv[i][0]->data = call.kv[i].self_k; kv[i][1]->data = call.kv[i].self_v;
            kv[i][2]->data = call.kv[i].cross_k; kv[i][3]->data = call.kv[i].cross_v;
        }
        for (auto * tensor : borrowed_views)
            tensor->data = static_cast<char *>(tensor->view_src->data) + tensor->view_offs;
    }
    void clear() {
        for (auto * tensor : borrowed_views) tensor->data = nullptr;
        for (auto * tensor : roots) tensor->data = nullptr;
    }
};
struct Decoder {
    cpu_decoder_config c;
    bool stock_q8;
    bool storage_api = false;
    ggml_type self_type = GGML_TYPE_F32, cross_type = GGML_TYPE_F32;
    std::string capture_prefix;
    bool capture_complete = false;
    ggml_backend_t backend = nullptr;
    ggml_threadpool_t pool = nullptr, cross_pool = nullptr;
    int cross_threads = 0;
    ggml_context * ctx = nullptr;
    ggml_context * dense_ctx = nullptr;
    ggml_backend_buffer_t dense_buffer = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_context * projection_ctx = nullptr;
    ggml_backend_buffer_t projection_buffer = nullptr;
    Range packed_projection_bytes = {};
    ggml_tensor * projection = nullptr;
    Range projection_bytes = {};
    ggml_context * token_ctx = nullptr;
    ggml_tensor * token_embeddings = nullptr, * token_positions = nullptr;
    cpu_decoder_token_plan token_plan = {};
    std::map<std::string, ggml_tensor *> weights;
    ggml_tensor * cross_weight = nullptr;
    std::array<std::unique_ptr<Graph>, 2> modes;
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
    explicit Decoder(cpu_decoder_config config, bool q8 = false): c(config), stock_q8(q8) {}
    ~Decoder() {
        modes[0].reset(); modes[1].reset();
        if (projection_buffer) ggml_backend_buffer_free(projection_buffer);
        if (projection_ctx) ggml_free(projection_ctx);
        if (token_ctx) ggml_free(token_ctx);
        if (dense_buffer) ggml_backend_buffer_free(dense_buffer);
        if (dense_ctx) ggml_free(dense_ctx);
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
        if (pool) ggml_threadpool_free(pool);
        if (cross_pool) ggml_threadpool_free(cross_pool);
    }
    void configure_storage(const cpu_decoder_storage_plan & plan) {
        require(plan.self_capacity == c.target && plan.cross_capacity == c.source,
                "resolved KV capacities differ from supported decoder geometry");
        self_type = storage_type(plan.self_encoding);
        cross_type = storage_type(plan.cross_encoding);
        storage_api = true;
    }
    void prepare(const cpu_encoder_weight * descriptors, size_t count) {
        require(c.width == 384 && c.layers == 4 && c.heads == 6 && c.ffn == 1536 && c.source == 1500 &&
                c.target == 448 && c.threads >= 1 && c.threads <= 64, "unsupported CPU decoder configuration");
        require(descriptors && count > 0 && count < 256, "invalid body weight descriptors");
        backend = ggml_backend_cpu_init(); require(backend, "CPU backend initialization failed");
        auto parameters = ggml_threadpool_params_default(c.threads); parameters.poll = 0;
        pool = ggml_threadpool_new(&parameters); require(pool, "CPU threadpool initialization failed");
        ggml_backend_cpu_set_n_threads(backend, c.threads); ggml_backend_cpu_set_threadpool(backend, pool);
        const size_t weight_metadata = (count + 2) * ggml_tensor_overhead();
        ctx = ggml_init({weight_metadata, nullptr, true});
        require(ctx, "body weight metadata allocation failed");
        if (stock_q8) {
            dense_ctx = ggml_init({weight_metadata, nullptr, true});
            require(dense_ctx, "Q8 body metadata allocation failed");
        }
        std::map<std::string, const cpu_encoder_weight *> source;
        for (size_t i = 0; i < count; ++i) {
            const auto & d = descriptors[i];
            require(d.name && d.rank >= 1 && d.rank <= 2, "invalid body weight descriptor");
            uint64_t bytes = 4;
            for (int a = 0; a < d.rank; ++a) {
                require(d.ne[a] > 0 && d.ne[a] <= 100000 && bytes <= UINT64_MAX / uint64_t(d.ne[a]), "invalid weight shape");
                bytes *= d.ne[a];
            }
            check_range({d.data, d.bytes}, bytes, alignof(float));
            require(source.emplace(d.name, &d).second, "duplicate body weight");
        }
        size_t consumed = 0;
        std::array<const cpu_encoder_weight *, 8> cross_sources = {};
        auto add = [&](const std::string & name, std::initializer_list<int64_t> shape) {
            auto it = source.find(name); require(it != source.end(), "missing body weight");
            const auto & d = *it->second; require(size_t(d.rank) == shape.size(), "body weight rank differs");
            int i = 0; for (auto n : shape) require(d.ne[i++] == n, "body weight shape differs");
            ++consumed; // Required names are unique in the fixed topology; source rejects duplicates.
            const int cross_index = cross_projection_index(name, c.layers);
            if (stock_q8 && cross_index >= 0) {
                cross_sources[cross_index] = &d;
                return;
            }
            const bool dense = stock_q8 && d.rank == 2;
            weights[name] = ggml_new_tensor(dense ? dense_ctx : ctx, dense ? GGML_TYPE_Q8_0 : GGML_TYPE_F32, d.rank, d.ne);
            ggml_set_name(weights[name], name.c_str());
        };
        auto linear = [&](const std::string & p, int in, int out) { add(p + ".weight", {in, out}); add(p + ".bias", {out}); };
        auto norm = [&](const std::string & p) { add(p + ".weight", {c.width}); add(p + ".bias", {c.width}); };
        norm("decoder.layer_norm");
        for (int i = 0; i < 4; ++i) {
            const std::string p = "decoder.layers." + std::to_string(i);
            for (const char * name : {"self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm"}) norm(p + "." + name);
            for (const char * name : {"self_attn", "encoder_attn"})
                for (const char * projection : {"q_proj", "k_proj", "v_proj", "out_proj"}) linear(p + "." + name + "." + projection, c.width, c.width);
            linear(p + ".fc1", c.width, c.ffn); linear(p + ".fc2", c.ffn, c.width);
        }
        require(consumed == source.size(), "unexpected body weight");
        if (stock_q8) {
            cross_weight = ggml_new_tensor_2d(dense_ctx, GGML_TYPE_Q8_0, c.width, 2*c.layers*c.width);
            ggml_set_name(cross_weight, "decoder.cross_projection.fused.weight");
        }
        if (stock_q8) {
            // Same ordered public extra-buft selection as whisper.cpp. Each
            // admitted provider must support every Q8 matrix/query geometry.
            auto * device = ggml_backend_get_device(backend);
            auto * registry = ggml_backend_dev_backend_reg(device);
            auto get_extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
                ggml_backend_reg_get_proc_address(registry, "ggml_backend_dev_get_extra_bufts"));
            auto * choices = get_extra ? get_extra(device) : nullptr;
            auto supports = [&](ggml_backend_buffer_type_t candidate, const std::string & name,
                                int64_t inside, int64_t outside, int queries) {
                auto * probe = ggml_init({3 * ggml_tensor_overhead(), nullptr, true});
                require(probe, "CPU Q8 provider probe metadata allocation failed");
                auto * weight = ggml_new_tensor_2d(probe, GGML_TYPE_Q8_0, inside, outside);
                auto * input = ggml_new_tensor_2d(probe, GGML_TYPE_F32, inside, queries);
                auto * op = ggml_mul_mat(probe, weight, input);
                weight->buffer = ggml_backend_buft_alloc_buffer(candidate, 0);
                const bool result = weight->buffer && ggml_backend_dev_supports_op(device, op);
                if (weight->buffer) ggml_backend_buffer_free(weight->buffer);
                weight->buffer = nullptr; ggml_free(probe);
                if (!result)
                    std::fprintf(stderr, "CPU_GGML_Q8_PROVIDER_REJECT role=decoder provider=%s weight=%s queries=%d rows=%lld\n",
                        ggml_backend_buft_name(candidate), name.c_str(), queries, (long long)outside);
                return result;
            };
            // Keep the original provider choice/domain. Fused-geometry refusal
            // restores its original eight matrices, never another quantizer.
            auto selected = mstar_cpu_ggml::first_supported_q8_provider(
                choices, ggml_backend_buft_name, [&](ggml_backend_buffer_type_t candidate) {
                    for (int queries : {1, 4, c.source}) for (const auto & entry : source) {
                        const auto & d = *entry.second;
                        if (d.rank == 2 && !supports(candidate, entry.first, d.ne[0], d.ne[1], queries)) return false;
                    }
                    return true;
                });
            require(selected, "actual CPU_KLEIDIAI/CPU_REPACK Q8 matrix support is unavailable");
            if (!supports(selected, "decoder.cross_projection.fused.weight", c.width, 2*c.layers*c.width, c.source)) {
                for (auto it = weights.begin(); it != weights.end();)
                    if (it->second->type == GGML_TYPE_Q8_0) it = weights.erase(it); else ++it;
                cross_weight = nullptr;
                ggml_free(dense_ctx);
                dense_ctx = ggml_init({weight_metadata, nullptr, true});
                require(dense_ctx, "unfused Q8 metadata allocation failed");
                for (const auto & entry : source) if (entry.second->rank == 2) {
                    const auto & d = *entry.second;
                    auto * tensor = ggml_new_tensor_2d(dense_ctx, GGML_TYPE_Q8_0, d.ne[0], d.ne[1]);
                    ggml_set_name(tensor, entry.first.c_str()); weights.emplace(entry.first, tensor);
                }
            }
            dense_buffer = ggml_backend_alloc_ctx_tensors_from_buft(dense_ctx, selected);
            require(dense_buffer, "Q8 provider weight allocation failed");
            ggml_backend_buffer_set_usage(dense_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            std::fprintf(stderr, "CPU_GGML_Q8_PROVIDER role=decoder provider=%s buffer=%s order=upstream_extra_bufts\n",
                         ggml_backend_buft_name(selected), ggml_backend_buffer_name(dense_buffer));
        }
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); require(buffer, "body weight allocation failed");
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        auto upload_q8 = [&](ggml_tensor * tensor, const float * data, size_t n) {
            // Same stock per-row F16 rounding and canonical Q8_0 conversion.
            std::vector<ggml_fp16_t> half(n); std::vector<float> rounded(n);
            ggml_cpu_fp32_to_fp16(data, half.data(), n);
            ggml_cpu_fp16_to_fp32(half.data(), rounded.data(), n);
            std::vector<uint8_t> canonical(ggml_nbytes(tensor));
            require(ggml_quantize_chunk(GGML_TYPE_Q8_0, rounded.data(), canonical.data(), 0,
                tensor->ne[1], tensor->ne[0], nullptr) == canonical.size(), "Q8 body packing size differs");
            ggml_backend_tensor_set(tensor, canonical.data(), 0, canonical.size());
        };
        for (const auto & entry : weights) {
            const auto & d = *source.at(entry.first); auto * tensor = entry.second;
            if (tensor->type == GGML_TYPE_F32) ggml_backend_tensor_set(tensor, d.data, 0, d.bytes);
            else upload_q8(tensor, d.data, d.bytes / sizeof(float));
        }
        if (cross_weight) {
            std::vector<float> joined(size_t(2*c.layers)*c.width*c.width);
            for (int layer = 0; layer < c.layers; ++layer) for (int part = 0; part < 2; ++part) {
                const size_t index = size_t(2*layer+part);
                const auto & weight = *cross_sources[index];
                std::memcpy(joined.data()+index*c.width*c.width,weight.data,weight.bytes);
            }
            upload_q8(cross_weight, joined.data(), joined.size());
        }
        std::fprintf(stderr, "CPU_GGML_CROSS_PROJECTION_READY matrices=%d fused_bias_set_rows=%d transient_joined_bytes=%llu\n",
            cross_weight ? 1 : 2*c.layers, cross_weight ? 1 : 0,
            (unsigned long long)(cross_weight ? uint64_t(2*c.layers)*c.width*c.source*sizeof(float) : 0));
        std::fprintf(stderr, "CPU_GGML_DECODER_WEIGHTS_READY q8=%d weights_bytes=%zu\n", stock_q8,
            ggml_backend_buffer_get_size(buffer) + (dense_buffer ? ggml_backend_buffer_get_size(dense_buffer) : 0));
    }
    void set_projection(const void * data, uint64_t bytes, int64_t rows, int64_t width, ggml_type type = GGML_TYPE_Q8_0, bool kleidiai = false) {
        require(type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0, "unsupported readout format");
        require(!kleidiai || type == GGML_TYPE_Q4_0, "KAI readout requires canonical Q4_0");
        require(storage_api && !projection && !modes[0] && !modes[1],
                "readout must be configured once on an idle storage handle before execution");
        require(width == c.width && width % 32 == 0 && rows > 0 && rows <= 100000,
                "readout logical shape differs");
        check_range({data, bytes}, uint64_t(rows) * ggml_row_size(type, width), 4);
        auto * metadata = ggml_init({ggml_tensor_overhead(), nullptr, true});
        require(metadata, "readout metadata allocation failed");
        projection_ctx = metadata;
        projection = ggml_new_tensor_2d(metadata, type, width, rows);
        if (kleidiai) {
            // Derived immutable weight packing remains in this prepared Decoder,
            // exactly like dense body weights. Canonical binding bytes stay live.
            auto * device = ggml_backend_get_device(backend);
            auto * registry = ggml_backend_dev_backend_reg(device);
            auto get_extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
                ggml_backend_reg_get_proc_address(registry, "ggml_backend_dev_get_extra_bufts"));
            auto * choices = get_extra ? get_extra(device) : nullptr;
            ggml_backend_buffer_type_t selected = nullptr;
            for (; choices && *choices; ++choices)
                if (std::strcmp(ggml_backend_buft_name(*choices), "CPU_KLEIDIAI") == 0) { selected = *choices; break; }
            require(selected, "CPU_KLEIDIAI Q4 readout provider unavailable; no fallback");
            auto * probe = ggml_init({3 * ggml_tensor_overhead(), nullptr, true});
            require(probe, "KAI readout probe metadata allocation failed");
            auto * weight = ggml_new_tensor_2d(probe, type, width, rows);
            auto * input = ggml_new_tensor_2d(probe, GGML_TYPE_F32, width, 1);
            auto * op = ggml_mul_mat(probe, weight, input);
            weight->buffer = ggml_backend_buft_alloc_buffer(selected, 0);
            const bool supported = weight->buffer && ggml_backend_dev_supports_op(device, op);
            if (weight->buffer) ggml_backend_buffer_free(weight->buffer);
            weight->buffer = nullptr; ggml_free(probe);
            require(supported, "CPU_KLEIDIAI Q4 readout shape unsupported; no fallback");
            projection_buffer = ggml_backend_alloc_ctx_tensors_from_buft(metadata, selected);
            require(projection_buffer, "KAI readout weight allocation failed");
            ggml_backend_buffer_set_usage(projection_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_tensor_set(projection, data, 0, bytes);
            packed_projection_bytes = {ggml_backend_buffer_get_base(projection_buffer), ggml_backend_buffer_get_size(projection_buffer)};
        } else projection->data = const_cast<void *>(data);
        ggml_set_name(projection, kleidiai ? "decoder.proj_out.weight.kleidiai_q4_0" : type == GGML_TYPE_Q8_0 ?
            "decoder.proj_out.weight.plain_q8_0" : "decoder.proj_out.weight.plain_q4_0");
        projection_bytes = {data, bytes};
        std::fprintf(stderr, "CPU_GGML_READOUT_READY provider=%s format=%s rows=%lld width=%lld threads=%d weight_bytes=%llu ownership=binding_immutable derived_packed_bytes=%llu\n",
            kleidiai ? "CPU_KLEIDIAI" : "CPU", type == GGML_TYPE_Q8_0 ? "Q8_0" : "Q4_0", (long long)rows, (long long)width, c.threads, (unsigned long long)bytes, (unsigned long long)packed_projection_bytes.bytes);
    }
    void set_token_inputs(const cpu_decoder_token_plan & plan) {
        require(storage_api && projection && !token_embeddings && !modes[0] && !modes[1],
                "token inputs require an idle configured readout handle before execution");
        const uint64_t rows = uint64_t(projection->ne[1]);
        check_range({plan.embeddings, plan.embedding_bytes}, rows * c.width * 4, 4);
        check_range({plan.positions, plan.position_bytes}, uint64_t(c.target) * c.width * 4, 4);
        require(plan.self_pages >= 2 && plan.cross_pages >= 2, "owner planes need sink/live capacity");
        require(std::isfinite(plan.embedding_scale) && plan.embedding_scale > 0.f, "invalid embedding scale");
        const uint64_t sb = uint64_t(plan.self_pages) * 2 * c.target * c.width * ggml_type_size(self_type);
        const uint64_t cb = uint64_t(plan.cross_pages) * 2 * c.source * c.width * ggml_type_size(cross_type);
        std::vector<Range> planes;
        for (const auto & layer : plan.layers) {
            check_range({layer.self_data, layer.self_bytes}, sb, ggml_type_size(self_type));
            check_range({layer.cross_data, layer.cross_bytes}, cb, ggml_type_size(cross_type));
            planes.push_back({layer.self_data, layer.self_bytes});
            planes.push_back({layer.cross_data, layer.cross_bytes});
        }
        const Range immutable[] = {{plan.embeddings, plan.embedding_bytes},
                                  {plan.positions, plan.position_bytes}, projection_bytes, packed_projection_bytes};
        for (size_t i = 0; i < planes.size(); ++i) {
            for (size_t j = 0; j < i; ++j) require(disjoint(planes[i], planes[j]), "owner planes overlap");
            for (auto other : immutable) require(disjoint(planes[i], other), "owner plane overlaps immutable weights");
        }
        token_ctx = ggml_init({2 * ggml_tensor_overhead(), nullptr, true});
        require(token_ctx, "token input metadata allocation failed");
        token_embeddings = ggml_new_tensor_2d(token_ctx, GGML_TYPE_F32, c.width, rows);
        token_positions = ggml_new_tensor_2d(token_ctx, GGML_TYPE_F32, c.width, c.target);
        token_embeddings->data = const_cast<float *>(plan.embeddings);
        token_positions->data = const_cast<float *>(plan.positions);
        token_plan = plan;
    }
    void validate(const cpu_decoder_storage_call & call) {
        require((call.queries == 1 || call.queries == 4) && call.self_length >= call.queries && call.self_length <= c.target &&
                call.cross_length >= 0 && call.cross_length <= c.source && call.kv && call.layer_count == 4, "invalid decoder call geometry");
        const int q = call.queries;
        check_range({call.kv, call.layer_count * sizeof(cpu_decoder_storage_kv)}, 4 * sizeof(cpu_decoder_storage_kv), alignof(cpu_decoder_storage_kv));
        std::vector<Range> read = {{call.x, call.x_bytes}, {call.self_positions, call.self_position_bytes},
                                 {call.self_mask, call.self_mask_bytes}, {call.cross_mask, call.cross_mask_bytes}};
        check_range(read[0], uint64_t(q) * c.width * 4, 4); check_range(read[1], uint64_t(q) * 8, 8);
        check_range(read[2], uint64_t(q) * c.target * 4, 4); check_range(read[3], uint64_t(q) * c.source * 4, 4);
        check_range({call.hidden, call.hidden_bytes}, uint64_t(c.width) * 4, 4);
        if (q == 4) {
            read.push_back({call.encoder, call.encoder_bytes}); read.push_back({call.cross_positions, call.cross_position_bytes});
            check_range(read[4], uint64_t(c.source) * c.width * 4, 4); check_range(read[5], uint64_t(c.source) * 8, 8);
        } else require(!call.encoder && !call.encoder_bytes && !call.cross_positions && !call.cross_position_bytes, "decode cannot rewrite cross KV");
        std::vector<Range> state;
        for (int i = 0; i < 4; ++i) {
            const auto & k = call.kv[i];
            const auto self_size = ggml_type_size(self_type), cross_size = ggml_type_size(cross_type);
            for (void * p : {k.self_k, k.self_v}) { check_range({p, k.self_bytes}, uint64_t(c.target) * c.width * self_size, self_size); state.push_back({p, k.self_bytes}); }
            for (void * p : {k.cross_k, k.cross_v}) { check_range({p, k.cross_bytes}, uint64_t(c.source) * c.width * cross_size, cross_size); state.push_back({p, k.cross_bytes}); }
        }
        for (size_t i = 0; i < state.size(); ++i) {
            for (size_t j = 0; j < i; ++j) require(disjoint(state[i], state[j]), "KV layer blocks overlap");
            for (auto r : read) require(disjoint(state[i], r), "borrowed call input overlaps KV storage");
            require(disjoint(state[i], {call.hidden, call.hidden_bytes}), "hidden output overlaps KV storage");
        }
        for (auto r : read) require(disjoint(r, {call.hidden, call.hidden_bytes}), "hidden output overlaps call input");
        positions_check(call.self_positions, q, c.target, call.self_length);
        if (q == 4) positions_check(call.cross_positions, c.source, c.source, call.cross_length);
        mask_check(call.self_mask, q, c.target, call.self_length);
        mask_check(call.cross_mask, q, c.source, call.cross_length);
    }
};

void validate_readout(Decoder & d, const cpu_decoder_readout_call & call) {
    require(d.projection, "readout entry point requires a prepared projection");
    const uint64_t rows = uint64_t(d.projection->ne[1]);
    Range logits{call.logits, call.logits_bytes}, token{call.token, call.token_bytes};
    check_range(logits, rows * 4, 4); check_range(token, 8, 8);
    require(call.suppressed_count <= rows, "suppression count exceeds vocabulary");
    Range suppression{call.suppressed, call.suppressed_bytes};
    if (call.suppressed_count) check_range(suppression, call.suppressed_count * 8, 8);
    else require(!call.suppressed_bytes, "empty suppression byte extent differs");
    for (uint64_t i = 0; i < call.suppressed_count; ++i)
        require(call.suppressed[i] >= 0 && uint64_t(call.suppressed[i]) < rows, "suppression index outside vocabulary");
    const auto & b = call.body;
    std::vector<Range> all = {{b.x, b.x_bytes}, {b.encoder, b.encoder_bytes},
        {b.self_positions, b.self_position_bytes}, {b.cross_positions, b.cross_position_bytes},
        {b.self_mask, b.self_mask_bytes}, {b.cross_mask, b.cross_mask_bytes},
        {b.hidden, b.hidden_bytes}, {b.kv, b.layer_count * sizeof(cpu_decoder_storage_kv)}, suppression};
    for (size_t i = 0; i < b.layer_count; ++i) {
        const auto & k = b.kv[i];
        for (void * data : {k.self_k, k.self_v}) all.push_back({data, k.self_bytes});
        for (void * data : {k.cross_k, k.cross_v}) all.push_back({data, k.cross_bytes});
    }
    require(disjoint(logits, token) && disjoint(logits, d.projection_bytes) && disjoint(token, d.projection_bytes) && disjoint(logits, d.packed_projection_bytes) && disjoint(token, d.packed_projection_bytes),
            "readout outputs overlap each other or projection");
    for (auto r : all) if (r.bytes) {
        require(disjoint(logits, r) && disjoint(token, r), "readout output overlaps borrowed storage");
        require(disjoint(d.projection_bytes, r) && disjoint(d.packed_projection_bytes, r), "projection overlaps mutable call storage");
    }
}
// Keep the old readout's finite-hidden rejection before activation quantization.
// One graph task copies only the final D384 vector; error is reported after
// synchronous completion, exactly as an old post-body readout error was.
void readout_input(ggml_tensor * dst, const ggml_tensor * src, int ith, int, void * userdata) {
    if (ith) return;
    auto & invalid = *static_cast<bool *>(userdata);
    const auto * x = static_cast<const float *>(src->data);
    invalid = false;
    for (int64_t i = 0; i < ggml_nelements(src); ++i) invalid |= !std::isfinite(x[i]);
    if (invalid) std::memset(dst->data, 0, ggml_nbytes(dst));
    else std::memcpy(dst->data, src->data, ggml_nbytes(dst));
}

void Graph::build(Decoder & owner, const cpu_decoder_storage_call & call) {
    const auto & c = owner.c;
    const int q = call.queries;
    ctx = ggml_init({max_nodes * ggml_tensor_overhead() + 3 * ggml_graph_overhead_custom(max_nodes, false), nullptr, true});
    require(ctx, "decoder graph metadata allocation failed");
    graph = ggml_new_graph_custom(ctx, max_nodes, false);
    auto borrow = [&](ggml_type type, int64_t n0, int64_t n1, const void * data) {
        auto * t = ggml_new_tensor_2d(ctx, type, n0, n1); t->data = const_cast<void *>(data); roots.push_back(t); return t;
    };
    auto linear = [&](const std::string & p, ggml_tensor * input) {
        return ggml_add(ctx, ggml_mul_mat(ctx, owner.weights.at(p + ".weight"), input), owner.weights.at(p + ".bias"));
    };
    auto norm = [&](const std::string & p, ggml_tensor * input) {
        return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, input, 1.e-5f), owner.weights.at(p + ".weight")), owner.weights.at(p + ".bias"));
    };
    auto attend = [&](ggml_tensor * query, ggml_tensor * key, ggml_tensor * value, int capacity, ggml_tensor * mask) {
        // Read the resolved plain-storage SET_ROWS aliases directly. Scale
        // stays in flash; no full-capacity conversion or scaled copy exists.
        const float scale = 1.f / std::sqrt(float(c.width / c.heads));
        auto split = [&](ggml_tensor * t, int n) { return ggml_permute(ctx, ggml_reshape_3d(ctx, t, c.width / c.heads, c.heads, n), 0, 2, 1, 3); };
        return ggml_reshape_2d(ctx, ggml_flash_attn_ext(ctx, split(query, q), split(key, capacity), split(value, capacity), mask,
                                                     scale, 0.f, 0.f), c.width, q);
    };
    ggml_tensor * input;
    if (owner.token_embeddings) {
        token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, q); ggml_set_input(token_ids);
        position_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, q); ggml_set_input(position_ids);
        auto * embedding = ggml_get_rows(ctx, owner.token_embeddings, token_ids);
        if (owner.token_plan.embedding_scale != 1.f)
            embedding = ggml_scale(ctx, embedding, owner.token_plan.embedding_scale);
        input = ggml_add(ctx, embedding, ggml_get_rows(ctx, owner.token_positions, position_ids));
    } else {
        x = borrow(GGML_TYPE_F32, c.width, q, call.x);
        input = x;
    }
    self_positions = borrow(GGML_TYPE_I64, q, 1, call.self_positions);
    self_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.target, q); ggml_set_input(self_mask);
    cross_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.source, q); ggml_set_input(cross_mask);
    if (q == 4) { encoder = borrow(GGML_TYPE_F32, c.width, c.source, call.encoder); cross_positions = borrow(GGML_TYPE_I64, c.source, 1, call.cross_positions); }
    ggml_tensor * projected_cross = nullptr;
    if (q == 4 && owner.cross_weight) {
        projected_cross = ggml_mul_mat(ctx, owner.cross_weight, encoder);
        ggml_set_name(projected_cross, "decoder.cross_projection.fused.raw");
    }
    // Keep one allocation/liveness graph. Its optional prefix performs all
    // cross writes before the token body; both plans borrow the same owners.
    std::array<std::array<ggml_tensor *, 2>, 4> cross;
    const bool separate_cross = q == 4 && owner.cross_pool;
    for (int i = 0; i < 4; ++i) {
        const auto & state = call.kv[i]; const std::string p = "decoder.layers." + std::to_string(i);
        kv[i] = {borrow(owner.self_type, c.width, c.target, state.self_k), borrow(owner.self_type, c.width, c.target, state.self_v),
                 borrow(owner.cross_type, c.width, c.source, state.cross_k), borrow(owner.cross_type, c.width, c.source, state.cross_v)};
        auto * ck = kv[i][2]; auto * cv = kv[i][3];
        if (q == 4) {
            auto projection = [&](int part, const char * name) {
                if (!projected_cross) return linear(p + ".encoder_attn." + name, encoder);
                return cross_projection_view(ctx,projected_cross,c.width,c.layers,c.source,i,part);
            };
            if (projected_cross) {
                ck = cross_bias_set_rows(ctx,ck,projection(0,"k_proj"),owner.weights.at(p+".encoder_attn.k_proj.bias"),cross_positions);
                cv = cross_bias_set_rows(ctx,cv,projection(1,"v_proj"),owner.weights.at(p+".encoder_attn.v_proj.bias"),cross_positions);
            } else {
                ck = ggml_set_rows(ctx,ck,projection(0,"k_proj"),cross_positions);
                cv = ggml_set_rows(ctx,cv,projection(1,"v_proj"),cross_positions);
            }
            if (separate_cross) {
                ggml_build_forward_expand(graph, ck);
                ggml_build_forward_expand(graph, cv);
            }
        }
        cross[i] = {ck, cv};
    }
    const int cross_nodes = ggml_graph_n_nodes(graph);
    auto * current = input;
    for (int i = 0; i < 4; ++i) {
        const std::string p = "decoder.layers." + std::to_string(i);
        auto * ck = cross[i][0]; auto * cv = cross[i][1];
        auto * n = norm(p + ".self_attn_layer_norm", current);
        auto * sq = linear(p + ".self_attn.q_proj", n);
        auto * sk = ggml_set_rows(ctx, kv[i][0], linear(p + ".self_attn.k_proj", n), self_positions);
        auto * sv = ggml_set_rows(ctx, kv[i][1], linear(p + ".self_attn.v_proj", n), self_positions);
        // Attention consumes the returned aliases, so every state write is a
        // real ancestor of the read. Base pointers alone would not order it.
        current = ggml_add(ctx, current, linear(p + ".self_attn.out_proj", attend(sq, sk, sv, c.target, self_mask)));
        n = norm(p + ".encoder_attn_layer_norm", current);
        auto * cq = linear(p + ".encoder_attn.q_proj", n);
        if (i == 0 && q == 1 && !owner.capture_prefix.empty()) {
            captured_query = cq; ggml_set_output(captured_query);
        }
        current = ggml_add(ctx, current, linear(p + ".encoder_attn.out_proj", attend(cq, ck, cv, c.source, cross_mask)));
        n = norm(p + ".final_layer_norm", current);
        current = ggml_add(ctx, current, linear(p + ".fc2", (owner.stock_q8 ? ggml_gelu(ctx, linear(p + ".fc1", n)) : ggml_gelu_erf(ctx, linear(p + ".fc1", n)))));
    }
    output = norm("decoder.layer_norm", current); ggml_set_output(output);
    if (owner.projection) {
        auto * last = ggml_view_2d(ctx, output, c.width, 1, output->nb[1], size_t(q - 1) * output->nb[1]);
        auto * finite = ggml_map_custom1(ctx, last, readout_input, 1, &invalid_readout_input);
        logits = ggml_mul_mat(ctx, owner.projection, finite);
        ggml_set_name(logits, "decoder.vocab.matmul"); ggml_set_output(logits);
    }
    ggml_build_forward_expand(graph, logits ? logits : output);
    std::set<ggml_tensor *> external(roots.begin(), roots.end());
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        if (node->view_src && external.count(node->view_src)) { external.insert(node); borrowed_views.push_back(node); }
        if (!owner.stock_q8 && node->op == GGML_OP_MUL_MAT) require(ggml_prec_set_acc(node, GGML_PREC_F32) && ggml_prec_set_src(node, GGML_PREC_F32, 1), "decoder matrix precision differs");
        if (node->op == GGML_OP_FLASH_ATTN_EXT) require(ggml_prec_set_acc(node, GGML_PREC_F32), "decoder attention precision differs");
        require(ggml_backend_supports_op(owner.backend, node), "unsupported CPU decoder graph operation");
    }
    allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(owner.backend));
    require(allocator && ggml_gallocr_alloc_graph(allocator, graph), "decoder graph allocation failed");
    body_graph = graph;
    if (separate_cross) {
        cross_graph = ggml_new_graph_custom(ctx, max_nodes, false);
        body_graph = ggml_new_graph_custom(ctx, max_nodes, false);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
            ggml_graph_add_node(i < cross_nodes ? cross_graph : body_graph, ggml_graph_node(graph, i));
        cross_plan = ggml_graph_plan(cross_graph, owner.cross_threads, owner.cross_pool);
        cross_scratch.resize(cross_plan.work_size); cross_plan.work_data = cross_scratch.data();
    }
    plan = ggml_graph_plan(body_graph, c.threads, owner.pool); scratch.resize(plan.work_size); plan.work_data = scratch.data();
    require(ggml_is_contiguous(output), "decoder hidden layout differs");
    // First-call allocation is deliberately inside the admitted timed call.
    std::fprintf(stderr, "CPU_GGML_DECODER_GRAPH_READY queries=%d workspace_bytes=%zu scratch_bytes=%zu borrowed_kv_bytes=%llu\n", q,
                 ggml_gallocr_get_buffer_size(allocator, 0), plan.work_size,
                 (unsigned long long)(uint64_t(4) * 2 * c.width *
                    (c.target * ggml_type_size(owner.self_type) + c.source * ggml_type_size(owner.cross_type))));
}
struct Busy {
    std::atomic_flag & flag;
    explicit Busy(std::atomic_flag & f): flag(f) { require(!flag.test_and_set(std::memory_order_acquire), "decoder workspace is busy"); }
    ~Busy() { flag.clear(std::memory_order_release); }
};
struct Clear { Graph & graph; ~Clear() { graph.clear(); } };
void write_capture(const std::string & path, const void * data, size_t size) {
    FILE * file = std::fopen(path.c_str(), "wbx");
    require(file, "capture output already exists or cannot be created");
    const bool written = std::fwrite(data, 1, size, file) == size;
    const bool closed = std::fclose(file) == 0;
    require(written && closed, "attention capture write failed");
}
void capture(Decoder & d, Graph & g, const cpu_decoder_storage_call & call) {
    if (d.capture_prefix.empty() || d.capture_complete || call.queries != 1) return;
    require(g.captured_query, "capture query is not retained");
    write_capture(d.capture_prefix + ".q.f32", g.captured_query->data, d.c.width * sizeof(float));
    std::vector<float> k(d.c.source * d.c.width), v(k.size());
    if (d.cross_type == GGML_TYPE_F32) {
        std::memcpy(k.data(), call.kv[0].cross_k, k.size() * sizeof(float));
        std::memcpy(v.data(), call.kv[0].cross_v, v.size() * sizeof(float));
    } else {
        ggml_cpu_fp16_to_fp32(static_cast<const ggml_fp16_t *>(call.kv[0].cross_k), k.data(), k.size());
        ggml_cpu_fp16_to_fp32(static_cast<const ggml_fp16_t *>(call.kv[0].cross_v), v.data(), v.size());
    }
    write_capture(d.capture_prefix + ".k.f32", k.data(), k.size() * sizeof(float));
    write_capture(d.capture_prefix + ".v.f32", v.data(), v.size() * sizeof(float));
    write_capture(d.capture_prefix + ".mask.f32", call.cross_mask, d.c.source * sizeof(float));
    char info[512];
    std::snprintf(info, sizeof(info),
        "{\"format\":\"actual-cross-attention-v1\",\"layer\":0,\"queries\":1,\"keys\":%d,"
        "\"width\":%d,\"heads\":%d,\"valid_keys\":%d,\"scale\":%.17g,\"storage\":\"%s\","
        "\"byte_order\":\"little\",\"values\":\"unscaled FP32 observation of actual operands\"}\n",
        d.c.source, d.c.width, d.c.heads, call.cross_length, double(1.f / std::sqrt(float(d.c.width / d.c.heads))),
        ggml_type_name(d.cross_type));
    write_capture(d.capture_prefix + ".json", info, std::strlen(info));
    d.capture_complete = true;
}
void run(Decoder & d, const cpu_decoder_storage_call & call, const cpu_decoder_readout_call * readout = nullptr,
         const cpu_decoder_token_call * tokens = nullptr) {
    require(bool(d.token_embeddings) == bool(tokens), "token entry differs from configured physical program");
    if (!tokens) d.validate(call);
    require(bool(d.projection) == bool(readout), "decoder/readout entry point differs from prepared handle");
    if (readout && !tokens) validate_readout(d, *readout);
    const int mode = call.queries == 4;
    if (!d.modes[mode]) { auto g = std::make_unique<Graph>(); g->build(d, call); d.modes[mode] = std::move(g); }
    auto & g = *d.modes[mode]; g.bind(call); Clear clear{g};
    // Dense masks contain only {0,-inf}; their F16 representation is exact.
    if (tokens) {
        auto * ids = static_cast<int32_t *>(g.token_ids->data);
        auto * positions = static_cast<int32_t *>(g.position_ids->data);
        auto * sm = static_cast<ggml_fp16_t *>(g.self_mask->data);
        auto * cm = static_cast<ggml_fp16_t *>(g.cross_mask->data);
        const auto zero = ggml_fp32_to_fp16(0.f), masked = ggml_fp32_to_fp16(-INFINITY);
        for (int row = 0; row < call.queries; ++row) {
            ids[row] = int32_t(tokens->tokens[row]); positions[row] = int32_t(tokens->positions[row]);
            for (int column = 0; column < d.c.target; ++column)
                sm[row * d.c.target + column] = column <= call.self_length - call.queries + row ? zero : masked;
            for (int column = 0; column < d.c.source; ++column)
                cm[row * d.c.source + column] = column < call.cross_length ? zero : masked;
        }
    } else {
        ggml_cpu_fp32_to_fp16(call.self_mask, static_cast<ggml_fp16_t *>(g.self_mask->data), call.queries * d.c.target);
        ggml_cpu_fp32_to_fp16(call.cross_mask, static_cast<ggml_fp16_t *>(g.cross_mask->data), call.queries * d.c.source);
    }
    if (g.cross_graph)
        require(ggml_graph_compute(g.cross_graph, &g.cross_plan) == GGML_STATUS_SUCCESS, "CPU cross preparation failed");
    require(ggml_graph_compute(g.body_graph, &g.plan) == GGML_STATUS_SUCCESS, "CPU decoder graph computation failed");
    std::memcpy(call.hidden, static_cast<float *>(g.output->data) + (call.queries - 1) * d.c.width, call.hidden_bytes);
    if (readout) {
        require(!g.invalid_readout_input, "readout hidden contains a nonfinite value");
        std::memcpy(readout->logits, g.logits->data, readout->logits_bytes);
        for (uint64_t i = 0; i < readout->suppressed_count; ++i)
            readout->logits[readout->suppressed[i]] = -INFINITY;
        int64_t best = 0;
        for (int64_t row = 1; row < d.projection->ne[1]; ++row)
            if ((!std::isnan(readout->logits[best]) && std::isnan(readout->logits[row])) ||
                    readout->logits[row] > readout->logits[best]) best = row;
        *readout->token = best;
    }
    if (!tokens) capture(d, g, call);
}
} // namespace
extern "C" {
int cpu_decoder_abi(void) { return 1; }
int cpu_decoder_q8_abi(void) { return 1; }
int cpu_decoder_q8_capability(void) {
    // Same public supports_op probe as upstream's weight-buft selection. No
    // model execution or model/request storage is created by discovery.
    auto * backend = ggml_backend_cpu_init();
    if (!backend) return 0;
    auto * device = ggml_backend_get_device(backend);
    auto get_extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(device), "ggml_backend_dev_get_extra_bufts"));
    auto * ctx = ggml_init({7 * ggml_tensor_overhead(), nullptr, true});
    if (!ctx) { ggml_backend_free(backend); return 0; }
    auto * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 32, 4);

    bool supported = false;
    for (auto * b = get_extra ? get_extra(device) : nullptr; b && *b; ++b) {
        if (std::strcmp(ggml_backend_buft_name(*b), "CPU_REPACK") != 0) continue;
        w->buffer = ggml_backend_buft_alloc_buffer(*b, 0);
        supported = w->buffer != nullptr;
        for (int q : {1, 4, 1500}) {
            auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, q);
            auto * op = ggml_mul_mat(ctx, w, x);
            supported = supported && ggml_backend_dev_supports_op(device, op);
        }
        if (w->buffer) ggml_backend_buffer_free(w->buffer);
        w->buffer = nullptr;
        break;
    }
    ggml_free(ctx); ggml_backend_free(backend);
    return supported ? 1 : 0;
}
void * cpu_decoder_create(const cpu_decoder_config * c, const cpu_encoder_weight * w, size_t n, char * error, size_t size) {
    try { require(c, "missing decoder configuration"); auto d = std::make_unique<Decoder>(*c); d->prepare(w, n); return d.release(); }
    catch (const std::exception & e) { message(error, size, e.what()); return nullptr; }
}
void * cpu_decoder_q8_create(const cpu_decoder_config * c, const cpu_encoder_weight * w, size_t n, char * error, size_t size) {
    try { require(c, "missing decoder configuration"); auto d = std::make_unique<Decoder>(*c, true); d->prepare(w, n); return d.release(); }
    catch (const std::exception & e) { message(error, size, e.what()); return nullptr; }
}
int cpu_decoder_storage_abi(void) { return 1; }
void * cpu_decoder_storage_create(const cpu_decoder_config * c, const cpu_decoder_storage_plan * plan,
        const cpu_encoder_weight * w, size_t n, char * error, size_t size) {
    try { require(c && plan, "missing storage configuration"); auto d = std::make_unique<Decoder>(*c);
        d->configure_storage(*plan); d->prepare(w, n); return d.release(); }
    catch (const std::exception & e) { message(error, size, e.what()); return nullptr; }
}
void * cpu_decoder_q8_storage_create(const cpu_decoder_config * c, const cpu_decoder_storage_plan * plan,
        const cpu_encoder_weight * w, size_t n, char * error, size_t size) {
    try { require(c && plan, "missing storage configuration"); auto d = std::make_unique<Decoder>(*c, true);
        d->configure_storage(*plan); d->prepare(w, n); return d.release(); }
    catch (const std::exception & e) { message(error, size, e.what()); return nullptr; }
}
int cpu_decoder_set_cross_threads(void * handle, int threads, char * error, size_t size) {
    try { require(handle, "missing decoder handle");
        auto & d = *static_cast<Decoder *>(handle); Busy busy(d.busy);
        require(threads >= 1 && threads <= 64, "invalid cross preparation thread count");
        require(!d.modes[0] && !d.modes[1] && !d.cross_pool, "cross threads must be set once before execution");
        auto parameters = ggml_threadpool_params_default(threads); parameters.poll = 0;
        d.cross_pool = ggml_threadpool_new(&parameters);
        require(d.cross_pool, "cross preparation threadpool initialization failed");
        d.cross_threads = threads; return 0;
    } catch (const std::exception & e) { message(error, size, e.what()); return -1; }
}
int cpu_decoder_capture_attention(void * handle, const char * prefix, char * error, size_t size) {
    try { require(handle && prefix && prefix[0] == '/', "capture requires an absolute file prefix");
        auto & d = *static_cast<Decoder *>(handle); Busy busy(d.busy);
        require(!d.modes[0] && d.capture_prefix.empty(), "capture must be armed once before Q1 preparation");
        d.capture_prefix = prefix; return 0; }
    catch (const std::exception & e) { message(error, size, e.what()); return -1; }
}
int cpu_decoder_run(void * handle, const cpu_decoder_call * call, char * error, size_t size) {
    try {
        require(handle && call, "missing decoder handle/call"); auto & d = *static_cast<Decoder *>(handle); Busy busy(d.busy);
        require(!d.storage_api, "legacy F32 entry point cannot run an encoded-storage handle");
        require(call->kv && call->layer_count == 4, "invalid legacy KV descriptors");
        check_range({call->kv, call->layer_count * sizeof(cpu_decoder_kv)},
                    4 * sizeof(cpu_decoder_kv), alignof(cpu_decoder_kv));
        std::array<cpu_decoder_storage_kv, 4> kv;
        for (size_t i = 0; i < kv.size(); ++i) { const auto & k = call->kv[i];
            kv[i] = {k.self_k, k.self_v, k.cross_k, k.cross_v, k.self_bytes, k.cross_bytes}; }
        const cpu_decoder_storage_call bound = {call->queries, call->self_length, call->cross_length,
            call->x, call->x_bytes, call->encoder, call->encoder_bytes,
            call->self_positions, call->self_position_bytes, call->cross_positions, call->cross_position_bytes,
            call->self_mask, call->self_mask_bytes, call->cross_mask, call->cross_mask_bytes,
            kv.data(), kv.size(), call->hidden, call->hidden_bytes};
        run(d, bound); return 0;
    } catch (const std::exception & e) { message(error, size, e.what()); return -1; }
}
int cpu_decoder_storage_run(void * handle, const cpu_decoder_storage_call * call, char * error, size_t size) {
    try { require(handle && call, "missing decoder handle/call");
        auto & d = *static_cast<Decoder *>(handle); Busy busy(d.busy);
        require(d.storage_api, "storage entry point requires a storage-ABI handle");
        run(d, *call); return 0;
    } catch (const std::exception & e) { message(error, size, e.what()); return -1; }
}
int cpu_decoder_readout_abi(void) { return 1; }
int cpu_decoder_set_readout_q8(void * handle, const void * data, uint64_t bytes,
        int64_t rows, int64_t width, char * error, size_t size) {
    try { require(handle, "missing decoder handle"); auto & d = *static_cast<Decoder *>(handle); Busy busy(d.busy);
        d.set_projection(data, bytes, rows, width); return 0;
    } catch (const std::exception & e) { message(error, size, e.what()); return -1; }
}
// Explicit preparation-only alternative. Existing Q8 setter/ABI are unchanged.
int cpu_decoder_set_readout_q4(void * handle, const void * data, uint64_t bytes,
        int64_t rows, int64_t width, char * error, size_t size) {
    try { require(handle, "missing decoder handle"); auto & d = *static_cast<Decoder *>(handle); Busy busy(d.busy);
        d.set_projection(data, bytes, rows, width, GGML_TYPE_Q4_0); return 0;
    } catch (const std::exception & e) { message(error, size, e.what()); return -1; }
}
int cpu_decoder_set_readout_q4_kai(void * handle, const void * data, uint64_t bytes,
        int64_t rows, int64_t width, char * error, size_t size) {
    try { require(handle, "missing decoder handle"); auto & d = *static_cast<Decoder *>(handle); Busy busy(d.busy);
        d.set_projection(data, bytes, rows, width, GGML_TYPE_Q4_0, true); return 0;
    } catch (const std::exception & e) { message(error, size, e.what()); return -1; }
}
int cpu_decoder_pack_readout_q4(const float * input, uint64_t input_bytes,
        void * output, uint64_t output_bytes, int64_t rows, int64_t width, char * error, size_t size) {
    try {
        require(rows > 0 && rows <= 100000 && width > 0 && width <= 16384 && width % 32 == 0,
                "invalid Q4 readout packing shape");
        const uint64_t count = uint64_t(rows)*uint64_t(width);
        const Range source = {input,input_bytes}, target = {output,output_bytes};
        check_range(source,count*sizeof(float),alignof(float));
        check_range(target,uint64_t(rows)*ggml_row_size(GGML_TYPE_Q4_0,width),4);
        require(disjoint(source,target), "Q4 readout packing input/output overlap");
        // Public model conversion rounds logical weights through F16 before
        // canonical Q4_0 quantization. Row-local scratch is preparation-only.
        for (uint64_t i = 0; i < count; ++i) require(std::isfinite(input[i]), "nonfinite Q4 readout weight");
        std::unique_ptr<ggml_fp16_t[]> half(new ggml_fp16_t[width]);
        std::unique_ptr<float[]> rounded(new float[width]);
        const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0,width);
        for (int64_t row = 0; row < rows; ++row) {
            ggml_cpu_fp32_to_fp16(input+row*width,half.get(),width);
            ggml_cpu_fp16_to_fp32(half.get(),rounded.get(),width);
            for (int64_t i = 0; i < width; ++i) require(std::isfinite(rounded[i]), "Q4 readout weight overflows F16");
            require(ggml_quantize_chunk(GGML_TYPE_Q4_0,rounded.get(),
                static_cast<char *>(output)+row*row_bytes,0,1,width,nullptr) == row_bytes,
                "canonical Q4 readout packing extent differs");
        }
        return 0;
    } catch (const std::exception & e) { message(error, size, e.what()); return -1; }
}
int cpu_decoder_storage_readout_run(void * handle, const cpu_decoder_readout_call * call, char * error, size_t size) {
    try { require(handle && call, "missing decoder readout handle/call");
        auto & d = *static_cast<Decoder *>(handle); Busy busy(d.busy);
        require(d.storage_api, "readout entry point requires a storage-ABI handle");
        run(d, call->body, call); return 0;
    } catch (const std::exception & e) { message(error, size, e.what()); return -1; }
}
int cpu_decoder_token_abi(void) { return 1; }
int cpu_decoder_set_token_inputs(void * handle, const cpu_decoder_token_plan * plan, char * error, size_t size) {
    try { require(handle && plan, "missing token input plan");
        auto & d = *static_cast<Decoder *>(handle); Busy busy(d.busy);
        d.set_token_inputs(*plan); return 0;
    } catch (const std::exception & e) { message(error, size, e.what()); return -1; }
}
int cpu_decoder_token_run(void * handle, const cpu_decoder_token_call * call, char * error, size_t size) {
    try { require(handle && call, "missing token decoder handle/call");
        auto & d = *static_cast<Decoder *>(handle); Busy busy(d.busy);
        require(d.token_embeddings && d.projection, "token input program was not prepared");
        const auto & c = d.c; const auto & p = d.token_plan; const auto & t = *call;
        const uint64_t rows = uint64_t(d.projection->ne[1]);
        require((t.queries == 1 || t.queries == 4) && t.self_length >= t.queries && t.self_length <= c.target &&
                t.self_begin == t.self_length - t.queries && t.cross_length >= 0 && t.cross_length <= c.source,
                "invalid admitted token geometry");
        require(t.self_page >= 0 && uint32_t(t.self_page) < p.self_pages && t.cross_page >= 0 &&
                uint32_t(t.cross_page) < p.cross_pages, "admitted page is outside owner plane");
        const Range ids{t.tokens, t.token_input_bytes}, positions{t.positions, t.position_input_bytes};
        const Range logits{t.logits, t.logits_bytes}, token{t.token, t.token_bytes};
        check_range(ids, uint64_t(t.queries) * 8, 8); check_range(positions, uint64_t(t.queries) * 8, 8);
        check_range(logits, rows * 4, 4); check_range(token, 8, 8);
        require(t.suppressed_count <= rows, "suppression count exceeds vocabulary");
        const Range suppression{t.suppressed, t.suppressed_bytes};
        if (t.suppressed_count) check_range(suppression, t.suppressed_count * 8, 8);
        else require(!t.suppressed_bytes, "empty suppression byte extent differs");
        std::vector<Range> inputs = {ids, positions};
        if (suppression.bytes) inputs.push_back(suppression);
        if (t.queries == 4) {
            check_range({t.encoder, t.encoder_bytes}, uint64_t(c.source) * c.width * 4, 4);
            require(t.cross_length == c.source, "full prefill cross write exceeds admitted prefix");
            inputs.push_back({t.encoder, t.encoder_bytes});
        } else require(!t.encoder && !t.encoder_bytes, "decode cannot rewrite cross KV");
        require(disjoint(logits, token), "token/logits outputs overlap");
        const Range immutable[] = {{p.embeddings, p.embedding_bytes}, {p.positions, p.position_bytes}, d.projection_bytes, d.packed_projection_bytes};
        for (auto r : immutable) {
            require(disjoint(logits, r) && disjoint(token, r), "output overlaps immutable weights");
        }
        for (auto r : inputs) require(disjoint(logits, r) && disjoint(token, r), "output overlaps input");
        for (const auto & layer : p.layers) {
            for (auto r : {Range{layer.self_data, layer.self_bytes}, Range{layer.cross_data, layer.cross_bytes}}) {
                require(disjoint(logits, r) && disjoint(token, r), "output overlaps owner plane");
                for (auto input : inputs) require(disjoint(r, input), "input overlaps owner plane");
            }
        }
        for (int i = 0; i < t.queries; ++i) {
            require(t.tokens[i] >= 0 && uint64_t(t.tokens[i]) < rows, "token ID is outside embedding table");
            require(t.positions[i] >= 0 && t.positions[i] < c.target, "position is outside embedding table");
        }
        for (uint64_t i = 0; i < t.suppressed_count; ++i)
            require(t.suppressed[i] >= 0 && uint64_t(t.suppressed[i]) < rows, "suppression index outside vocabulary");
        std::array<cpu_decoder_storage_kv, 4> kv;
        const uint64_t sb = uint64_t(c.target) * c.width * ggml_type_size(d.self_type);
        const uint64_t cb = uint64_t(c.source) * c.width * ggml_type_size(d.cross_type);
        for (int i = 0; i < 4; ++i) {
            auto * sk = static_cast<char *>(p.layers[i].self_data) + uint64_t(t.self_page) * 2 * sb;
            auto * ck = static_cast<char *>(p.layers[i].cross_data) + uint64_t(t.cross_page) * 2 * cb;
            kv[i] = {sk, sk + sb, ck, ck + cb, sb, cb};
        }
        std::array<int64_t, 4> self_positions;
        for (int i = 0; i < t.queries; ++i) self_positions[i] = t.self_begin + i;
        std::array<int64_t, 1500> cross_positions;
        if (t.queries == 4) for (int i = 0; i < c.source; ++i) cross_positions[i] = i;
        std::array<float, 384> hidden;
        cpu_decoder_storage_call body = {t.queries, t.self_length, t.cross_length,
            nullptr, 0, t.encoder, t.encoder_bytes, self_positions.data(), uint64_t(t.queries) * 8,
            t.queries == 4 ? cross_positions.data() : nullptr, t.queries == 4 ? uint64_t(c.source) * 8 : 0,
            nullptr, 0, nullptr, 0, kv.data(), kv.size(), hidden.data(), sizeof(hidden)};
        cpu_decoder_readout_call readout = {body, t.logits, t.logits_bytes, t.token, t.token_bytes,
                                         t.suppressed, t.suppressed_count, t.suppressed_bytes};
        run(d, body, &readout, &t); return 0;
    } catch (const std::exception & e) { message(error, size, e.what()); return -1; }
}
int cpu_decoder_close(void * handle, char * error, size_t size) {
    try { if (handle) { auto * d = static_cast<Decoder *>(handle); require(!d->busy.test_and_set(std::memory_order_acquire), "cannot close a busy decoder"); delete d; } return 0; }
    catch (const std::exception & e) { message(error, size, e.what()); return -1; }
}
}
