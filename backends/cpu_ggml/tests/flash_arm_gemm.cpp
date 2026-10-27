// Compile twice with the original and patched pinned header. Each translation
// unit uses the exact same SIMD FMA helper and independently named entry point.
#include "common.h"
#include MSTAR_GEMM_HEADER
extern "C" void MSTAR_GEMM_FUNCTION(float * c, const float * a, const float * b,
                                   int m, int k, int n) {
    simd_gemm(c, a, b, m, k, n);
}
extern "C" int MSTAR_GEMM_SHAPE(void) { return 100*GEMM_RM + GEMM_RN; }
