#pragma once
#include "decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Version 1 explicitly supports plain, unscaled NHD F32/F16 storage. These
 * identifiers describe this ABI's encodings, not a universal format registry.
 * New encodings require an explicit ABI extension, never dtype reinterpretation. */
enum cpu_decoder_plain_encoding {
    CPU_DECODER_PLAIN_F32 = 1,
    CPU_DECODER_PLAIN_F16 = 2,
};
struct cpu_decoder_storage_plan {
    uint32_t self_encoding, cross_encoding;
    int32_t self_capacity, cross_capacity;
};
struct cpu_decoder_storage_kv {
    void * self_k, * self_v, * cross_k, * cross_v;
    uint64_t self_bytes, cross_bytes;
};
struct cpu_decoder_storage_call {
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
    const struct cpu_decoder_storage_kv * kv;
    uint64_t layer_count;
    float * hidden;
    uint64_t hidden_bytes;
};

CPU_ENCODER_API int cpu_decoder_storage_abi(void);
CPU_ENCODER_API void * cpu_decoder_storage_create(const struct cpu_decoder_config *,
    const struct cpu_decoder_storage_plan *, const struct cpu_encoder_weight *,
    size_t count, char * error, size_t error_size);
CPU_ENCODER_API void * cpu_decoder_q8_storage_create(const struct cpu_decoder_config *,
    const struct cpu_decoder_storage_plan *, const struct cpu_encoder_weight *,
    size_t count, char * error, size_t error_size);
/* All storage is borrowed for this synchronous call, just as in decoder.h.
 * A legacy float-pointer entry point cannot run a storage-ABI handle. */
CPU_ENCODER_API int cpu_decoder_storage_run(void *, const struct cpu_decoder_storage_call *,
    char * error, size_t error_size);

/* Diagnostic only: capture the first Q1 layer-0 cross-attention operands.
 * prefix is an absolute, fresh file prefix; the caller creates its directory.
 * Must be armed before any Q1 graph exists. No capture is enabled by default. */
CPU_ENCODER_API int cpu_decoder_capture_attention(void *, const char * prefix,
    char * error, size_t error_size);

#ifdef __cplusplus
}
#endif
