// Actual pinned CPU kernels and graph allocation, including residual lifetime.
#include "../encoder.cpp"
#include <cassert>
#include <limits>

int main() {
    int executions = 0;
    for (int rows : {1, 4, 17, 1500}) for (int threads : {1, 2, 4}) for (int stride : {1, 2}) {
        constexpr int width = 384;
        auto * ctx = ggml_init({2u << 20, nullptr, true}); assert(ctx);
        auto * backend = ggml_backend_cpu_init(); assert(backend);
        auto * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width*stride, rows); ggml_set_input(storage);
        auto * input = ggml_view_2d(ctx, storage, width, rows, storage->nb[1], 0);
        auto * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width); ggml_set_input(weight);
        auto * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width); ggml_set_input(bias);
        // Production weights are externally allocated and cannot be recycled
        // by gallocr. Keep the fixture's gallocr-owned leaves equally live.
        ggml_set_output(storage); ggml_set_output(weight); ggml_set_output(bias);
        auto * expected = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, input, 1.e-5f), weight), bias); ggml_set_output(expected);
        auto * actual = ggml_map_custom3(ctx, input, weight, bias, encoder_norm_affine, GGML_N_TASKS_MAX, nullptr); ggml_set_output(actual);
        // Preserve the actual residual input after the CUSTOM consumer, too.
        auto * residual = ggml_add(ctx, actual, input); ggml_set_output(residual);
        auto * graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, expected); ggml_build_forward_expand(graph, residual);
        assert(actual->src[0] == input && actual->src[1] == weight && actual->src[2] == bias && !actual->view_src);
        auto * allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)); assert(ggml_gallocr_alloc_graph(allocator, graph));
        assert(actual->data != input->data && actual->data != expected->data && actual->data != weight->data && actual->data != bias->data);
        auto params = ggml_threadpool_params_default(threads); params.poll = 0; auto * pool = ggml_threadpool_new(&params); assert(pool);
        auto plan = ggml_graph_plan(graph, threads, pool);
        std::vector<uint8_t> scratch(plan.work_size + 128, 0xa5); plan.work_data = scratch.data() + 64;
        for (int repeat = 0; repeat < 4; ++repeat) {
            std::vector<float> values(size_t(width)*stride*rows), w(width), b(width);
            for (size_t i = 0; i < values.size(); ++i) values[i] = float(int((i*13 + repeat*17)%127) - 63)/8;
            for (int i = 0; i < width; ++i) { w[i] = float((i*7 + repeat*11)%17 - 8)/32; b[i] = float((i*11 + repeat*3)%19 - 9)/64; }
            if (repeat == 2) { std::fill(values.begin(), values.begin()+width, -0.f); w[0] = -0.f; b[0] = -0.f; values[width-1] = std::numeric_limits<float>::denorm_min(); }
            if (repeat == 3) {
                values[0] = std::numeric_limits<float>::infinity(); values[1] = -std::numeric_limits<float>::infinity();
                uint32_t bits = 0xffc12345; std::memcpy(&values[2], &bits, 4); bits = 0x7f812345; std::memcpy(&values[3], &bits, 4);
                w[4] = std::numeric_limits<float>::quiet_NaN(); b[5] = -std::numeric_limits<float>::infinity();
            }
            ggml_backend_tensor_set(storage, values.data(), 0, values.size()*sizeof(float));
            ggml_backend_tensor_set(weight, w.data(), 0, w.size()*sizeof(float)); ggml_backend_tensor_set(bias, b.data(), 0, b.size()*sizeof(float));
            assert(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);
            assert(std::memcmp(actual->data, expected->data, ggml_nbytes(actual)) == 0);
            assert(std::memcmp(storage->data, values.data(), ggml_nbytes(storage)) == 0);
            assert(std::memcmp(weight->data, w.data(), ggml_nbytes(weight)) == 0 && std::memcmp(bias->data, b.data(), ggml_nbytes(bias)) == 0);
            for (size_t i = 0; i < 64; ++i) assert(scratch[i] == 0xa5 && scratch[64+plan.work_size+i] == 0xa5);
            ++executions;
        }
        ggml_threadpool_free(pool); ggml_gallocr_free(allocator); ggml_free(ctx); ggml_backend_free(backend);
    }
    std::printf("{\"status\":\"ENCODER_NORM_AFFINE_PASS\",\"executions\":%d,\"bitwise_equal\":true}\n", executions);
}
