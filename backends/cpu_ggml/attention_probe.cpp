#include "attention_probe.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
constexpr int width = 384, keys = 1500, heads = 6, dimension = 64;
void require(bool value, const char * reason) { if (!value) throw std::runtime_error(reason); }
struct Operator {
    ggml_context * context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cplan plan = {};
    std::vector<uint8_t> scratch;
    ~Operator() { if (context) ggml_free(context); }
    Operator(const cpu_attention_operands & input, ggml_type type, ggml_threadpool_t pool) {
        try {
            context = ggml_init({16u << 20, nullptr, false});
            require(context, "attention metadata allocation failed");
            auto * q = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, 1);
            auto * k = ggml_new_tensor_2d(context, type, width, keys);
            auto * v = ggml_new_tensor_2d(context, type, width, keys);
            auto * mask = ggml_new_tensor_2d(context, GGML_TYPE_F16, keys, 1);
            std::memcpy(q->data, input.q, width * sizeof(float));
            if (type == GGML_TYPE_F32) {
                std::memcpy(k->data, input.k, width * keys * sizeof(float));
                std::memcpy(v->data, input.v, width * keys * sizeof(float));
            } else {
                ggml_cpu_fp32_to_fp16(input.k, static_cast<ggml_fp16_t *>(k->data), width * keys);
                ggml_cpu_fp32_to_fp16(input.v, static_cast<ggml_fp16_t *>(v->data), width * keys);
            }
            ggml_cpu_fp32_to_fp16(input.mask, static_cast<ggml_fp16_t *>(mask->data), keys);
            auto split = [&](ggml_tensor * t, int n) {
                return ggml_permute(context, ggml_reshape_3d(context, t, dimension, heads, n), 0, 2, 1, 3);
            };
            output = ggml_flash_attn_ext(context, split(q, 1), split(k, keys), split(v, keys), mask,
                                          1.f / std::sqrt(float(dimension)), 0.f, 0.f);
            require(ggml_prec_set_acc(output, GGML_PREC_F32), "attention accumulation unsupported");
            graph = ggml_new_graph(context); ggml_build_forward_expand(graph, output);
            plan = ggml_graph_plan(graph, input.threads, pool);
            scratch.resize(plan.work_size); plan.work_data = scratch.data();
        } catch (...) { if (context) ggml_free(context); context = nullptr; throw; }
    }
    void run() { require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS, "attention execution failed"); }
};
template<class F> double timed(F work) {
    const auto before = std::chrono::steady_clock::now(); work();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - before).count();
}
struct Pool {
    ggml_threadpool_t pool = nullptr;
    explicit Pool(int count) { auto config = ggml_threadpool_params_default(count); config.poll = 0;
        pool = ggml_threadpool_new(&config); require(pool, "attention threadpool allocation failed"); }
    ~Pool() { if (pool) ggml_threadpool_free(pool); }
};
}
extern "C" {
int cpu_attention_probe_abi(void) { return 1; }
int cpu_attention_probe(const cpu_attention_operands * input, cpu_attention_sample * samples,
        float * output_f32, float * output_f16, char * error, size_t error_size) {
    try {
        require(input && samples && output_f32 && output_f16, "missing probe storage");
        require(input->q && input->k && input->v && input->mask &&
                input->q_bytes == width * sizeof(float) && input->kv_bytes == width * keys * sizeof(float) &&
                input->mask_bytes == keys * sizeof(float), "probe requires exact Q1/H6/D64/N1500 operands");
        require(input->threads >= 1 && input->threads <= 64 && input->warmup >= 1 && input->warmup <= 100 &&
                input->repetitions >= 1 && input->repetitions <= 10000, "invalid probe repetition/thread limits");
        for (int i = 0; i < keys; ++i)
            require(input->mask[i] == 0.f || input->mask[i] == -INFINITY, "probe mask must be zero/-infinity");
        Pool pool(input->threads);
        Operator f32(*input, GGML_TYPE_F32, pool.pool), f16(*input, GGML_TYPE_F16, pool.pool);
        std::vector<float> copied(2 * width * keys);
        std::vector<ggml_fp16_t> encoded(2 * width * keys);
        for (int i = 0; i < input->warmup; ++i) { f32.run(); f16.run(); }
        for (int i = 0; i < input->repetitions; ++i) {
            auto & sample = samples[i];
            if (i % 2) { sample.f16_ms = timed([&] { f16.run(); }); sample.f32_ms = timed([&] { f32.run(); }); }
            else { sample.f32_ms = timed([&] { f32.run(); }); sample.f16_ms = timed([&] { f16.run(); }); }
            auto encode = [&] {
                ggml_cpu_fp32_to_fp16(input->k, encoded.data(), width * keys);
                ggml_cpu_fp32_to_fp16(input->v, encoded.data() + width * keys, width * keys);
                asm volatile("" : : "g"(encoded.data()) : "memory");
            };
            auto copy = [&] {
                std::memcpy(copied.data(), input->k, input->kv_bytes);
                std::memcpy(copied.data() + width * keys, input->v, input->kv_bytes);
                asm volatile("" : : "g"(copied.data()) : "memory");
            };
            if (i % 2) { sample.copy_f32_ms = timed(copy); sample.encode_f16_ms = timed(encode); }
            else { sample.encode_f16_ms = timed(encode); sample.copy_f32_ms = timed(copy); }
        }
        // Consume written destinations, keeping the separately timed writes real.
        volatile float retained = copied.back() + ggml_fp16_to_fp32(encoded.back()); (void) retained;
        std::memcpy(output_f32, f32.output->data, width * sizeof(float));
        std::memcpy(output_f16, f16.output->data, width * sizeof(float));
        return 0;
    } catch (const std::exception & e) {
        if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
        return -1;
    }
}
}
