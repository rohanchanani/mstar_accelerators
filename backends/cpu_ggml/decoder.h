#pragma once
#include "encoder.h"
#ifdef __cplusplus
extern "C" {
#endif
struct cpu_decoder_config {
    int32_t width, layers, heads, ffn, source, target, threads;
};
/* Each block is the complete admitted DenseSequenceView layer, contiguous NHD.
 * The runtime owns these allocations, their identities, and valid prefixes. */
struct cpu_decoder_kv {
    float * self_k, * self_v, * cross_k, * cross_v;
    uint64_t self_bytes, cross_bytes;
};
struct cpu_decoder_call {
    int32_t queries, self_length, cross_length;
    const float * x;
    uint64_t x_bytes;
    const float * encoder;
    uint64_t encoder_bytes;
    const int64_t * self_positions;
    uint64_t self_position_bytes;
    const int64_t * cross_positions;
    uint64_t cross_position_bytes;
    const float * self_mask;
    uint64_t self_mask_bytes;
    const float * cross_mask;
    uint64_t cross_mask_bytes;
    const struct cpu_decoder_kv * kv;
    uint64_t layer_count;
    float * hidden;
    uint64_t hidden_bytes;
};
CPU_ENCODER_API int cpu_decoder_abi(void);
CPU_ENCODER_API int cpu_decoder_q8_abi(void);
CPU_ENCODER_API int cpu_decoder_q8_capability(void);
CPU_ENCODER_API void * cpu_decoder_q8_create(const struct cpu_decoder_config *,
    const struct cpu_encoder_weight *, size_t count, char * error, size_t error_size);
CPU_ENCODER_API void * cpu_decoder_create(const struct cpu_decoder_config *,
    const struct cpu_encoder_weight *, size_t count, char * error, size_t error_size);
/* Borrows every pointer for this synchronous call only; clears bindings after
 * threadpool completion. The final D384 hidden vector is caller-owned. */
CPU_ENCODER_API int cpu_decoder_run(void *, const struct cpu_decoder_call *, char *, size_t);
/* Optional physical preparation policy, set once before either mode executes.
 * Cross writes finish before the token body; cache ownership is unchanged. */
CPU_ENCODER_API int cpu_decoder_set_cross_threads(void *, int, char *, size_t);
CPU_ENCODER_API int cpu_decoder_close(void *, char *, size_t);
#ifdef __cplusplus
}
#endif
