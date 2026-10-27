#pragma once
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define CPU_ENCODER_API __declspec(dllexport)
#else
#define CPU_ENCODER_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif

/* Physical CPU ABI only. The caller owns all logical weights and call storage. */
struct cpu_encoder_config {
    int32_t width, layers, heads, ffn, mels, positions, threads;
};
struct cpu_encoder_weight {
    const char * name;
    const float * data;
    uint64_t bytes;
    int32_t rank;
    int64_t ne[3]; /* ggml fastest-first; contiguous F32 */
};
struct cpu_encoder_memory {
    uint64_t weights, graph, metadata, scratch;
};

CPU_ENCODER_API int cpu_encoder_abi(void);
CPU_ENCODER_API int cpu_encoder_q8_abi(void);
CPU_ENCODER_API int cpu_encoder_q8_capability(void);
CPU_ENCODER_API void * cpu_encoder_create(const struct cpu_encoder_config *,
    const struct cpu_encoder_weight *, size_t count, struct cpu_encoder_memory *,
    char * error, size_t error_size);
CPU_ENCODER_API void * cpu_encoder_q8_create(const struct cpu_encoder_config *,
    const struct cpu_encoder_weight *, size_t count, struct cpu_encoder_memory *,
    char * error, size_t error_size);
/* Synchronous: inputs are borrowed only until return. Output is caller-owned. */
CPU_ENCODER_API int cpu_encoder_run(void *, const float * features, uint64_t feature_bytes,
    float * states, uint64_t state_bytes, char * error, size_t error_size);
/* The outer workspace owner must drain before close. Busy close is rejected. */
CPU_ENCODER_API int cpu_encoder_close(void *, char * error, size_t error_size);

#ifdef __cplusplus
}
#endif
