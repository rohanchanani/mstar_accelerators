#pragma once
#include "encoder.h"
#ifdef __cplusplus
extern "C" {
#endif
struct cpu_attention_operands {
    const float * q, * k, * v, * mask;
    uint64_t q_bytes, kv_bytes, mask_bytes;
    int32_t threads, warmup, repetitions;
};
struct cpu_attention_sample {
    double f32_ms, f16_ms, encode_f16_ms, copy_f32_ms;
};
/* Bounded operator measurement: Q1/H6/D64/N1500, unscaled K, identical mask,
 * scale1/sqrt64. The caller supplies captured actual operands. Preparation and
 * encoding are excluded from attention timings; encoding/write cost is separate.
 * No model, serving, ownership, or numerical qualification claim. */
CPU_ENCODER_API int cpu_attention_probe_abi(void);
CPU_ENCODER_API int cpu_attention_probe(const struct cpu_attention_operands *,
    struct cpu_attention_sample *, float * output_f32, float * output_f16,
    char * error, size_t error_size);
#ifdef __cplusplus
}
#endif
