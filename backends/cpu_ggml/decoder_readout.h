#pragma once
#include "decoder_storage.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Optional plain-Q8 vocabulary projection, inside the existing decoder graph.
 * Packed weights are immutable binding-owned Q8_0 bytes and MUST remain alive
 * until cpu_decoder_close completes. This setter is preparation-only.
 * No embedding, KV, sampler or request ownership is transferred. */
CPU_ENCODER_API int cpu_decoder_readout_abi(void);
CPU_ENCODER_API int cpu_decoder_set_readout_q8(void *, const void *, uint64_t,
    int64_t rows, int64_t width, char *, size_t);
/* Explicit Q4_0 alternative with the same borrowed-weight lifetime. The packer
 * uses the pinned public F16-rounding + canonical Q4_0 quantizer. It owns no
 * storage and is called only during physical preparation, never execution. */
CPU_ENCODER_API int cpu_decoder_pack_readout_q4(const float *, uint64_t,
    void *, uint64_t, int64_t rows, int64_t width, char *, size_t);
CPU_ENCODER_API int cpu_decoder_set_readout_q4(void *, const void *, uint64_t,
    int64_t rows, int64_t width, char *, size_t);
/* Explicit KAI Q4 alternative. Copies/repackages these same canonical bytes
 * into the existing prepared decoder's immutable weight buffer. The canonical
 * borrowed lifetime is retained; graphs are destroyed before derived storage.
 * Unsupported actual provider/shape rejects before execution, never falls back. */
CPU_ENCODER_API int cpu_decoder_set_readout_q4_kai(void *, const void *, uint64_t,
    int64_t rows, int64_t width, char *, size_t);
struct cpu_decoder_readout_call {
    struct cpu_decoder_storage_call body;
    float * logits;
    uint64_t logits_bytes;
    int64_t * token;
    uint64_t token_bytes;
    const int64_t * suppressed;
    uint64_t suppressed_count, suppressed_bytes;
};
/* Validates every borrowed range and suppression index before any KV write.
 * Logits/token are caller-owned. Suppression and first-index argmax match the
 * existing ARM readout; positive sampling remains with its current owner. */
CPU_ENCODER_API int cpu_decoder_storage_readout_run(void *,
    const struct cpu_decoder_readout_call *, char *, size_t);
#ifdef __cplusplus
}
#endif
