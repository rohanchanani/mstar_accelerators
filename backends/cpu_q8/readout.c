/* Synchronous CPU operation probe. No Torch ABI, heap, cache or thread pool. */
#include <asm/hwcap.h>
#include <sys/auxv.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include "quants.h"

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_DOTPROD)
#error "This explicit probe requires AArch64 dot-product compilation"
#endif
#define EXPORT __attribute__((visibility("default")))
_Static_assert(sizeof(block_q8_0) == 34, "standard Q8_0 block extent required");

EXPORT int q8cpu_capability(void) {
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
}

EXPORT int q8cpu_quantize(const float *x, void *packed, size_t bytes, int64_t k) {
    if (!q8cpu_capability()) return ENOTSUP;
    if (!x || !packed || k <= 0 || k > 4096 || k % 32 || bytes != (size_t)(k / 32) * sizeof(block_q8_0)) return EINVAL;
    for (int64_t i = 0; i < k; ++i) if (!isfinite(x[i])) return EDOM;
    quantize_row_q8_0(x, packed, k);
    return 0;
}

EXPORT int q8cpu_readout(const void *weights, size_t bytes, int64_t rows, int64_t k,
                        const float *x, float *out, int64_t *token,
                        const int64_t *suppressed, size_t suppressed_count) {
    if (!q8cpu_capability()) return ENOTSUP;
    if (!weights || !x || !out || !token || rows <= 0 || k <= 0 || k > 4096 || k % 32 ||
        (suppressed_count && !suppressed)) return EINVAL;
    const size_t row_bytes = (size_t)(k / 32) * sizeof(block_q8_0);
    if ((uint64_t)rows > SIZE_MAX / row_bytes || bytes != (size_t)rows * row_bytes) return EINVAL;
    for (size_t i = 0; i < suppressed_count; ++i) if (suppressed[i] < 0 || suppressed[i] >= rows) return EINVAL;
    /* The actual K384 call uses 12*34=408 stack bytes. All mutable output is
       caller-owned, and no pointer survives this synchronous call. */
    _Alignas(16) block_q8_0 activation[k / 32];
    int status = q8cpu_quantize(x, activation, row_bytes, k);
    if (status) return status;
    for (int64_t row = 0; row < rows; ++row) {
        ggml_vec_dot_q8_0_q8_0((int)k, out + row, 0,
            (const uint8_t *)weights + (size_t)row * row_bytes, 0, activation, 0, 1);
    }
    for (size_t i = 0; i < suppressed_count; ++i) out[suppressed[i]] = -INFINITY;
    int64_t best = 0;
    for (int64_t row = 1; row < rows; ++row) {
        /* Match first-index ties and first NaN, including signed zero/all-inf. */
        if ((!isnan(out[best]) && isnan(out[row])) || out[row] > out[best]) best = row;
    }
    *token = best;
    return 0;
}
