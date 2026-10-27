#pragma once
#include "decoder_readout.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Optional token entry on the existing storage/readout handle. All planes and
 * immutable tables remain borrowed from the binding/resource until close.
 * Plane layout is [page, K/V, time, head, dimension], with no private cache. */
struct cpu_decoder_token_plane {
    void * self_data; uint64_t self_bytes;
    void * cross_data; uint64_t cross_bytes;
};
struct cpu_decoder_token_plan {
    const float * embeddings; uint64_t embedding_bytes;
    const float * positions; uint64_t position_bytes;
    uint32_t self_pages, cross_pages;
    float embedding_scale;
    struct cpu_decoder_token_plane layers[4];
};
struct cpu_decoder_token_call {
    int32_t queries, self_page, cross_page, self_length, cross_length, self_begin;
    const int64_t * tokens; uint64_t token_input_bytes;
    const int64_t * positions; uint64_t position_input_bytes;
    const float * encoder; uint64_t encoder_bytes;
    float * logits; uint64_t logits_bytes;
    int64_t * token; uint64_t token_bytes;
    const int64_t * suppressed; uint64_t suppressed_count, suppressed_bytes;
};
CPU_ENCODER_API int cpu_decoder_token_abi(void);
CPU_ENCODER_API int cpu_decoder_set_token_inputs(void *, const struct cpu_decoder_token_plan *, char *, size_t);
/* Current plans supply page/length/write metadata; no plan/request is retained.
 * All ranges, indices and input IDs are validated before any cache write.
 * Positive sampling consumes caller-owned logits through its existing owner. */
CPU_ENCODER_API int cpu_decoder_token_run(void *, const struct cpu_decoder_token_call *, char *, size_t);
#ifdef __cplusplus
}
#endif
