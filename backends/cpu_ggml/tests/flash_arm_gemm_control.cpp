#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
extern "C" void original_gemm(float *, const float *, const float *, int, int, int);
extern "C" void patched_gemm(float *, const float *, const float *, int, int, int);
extern "C" int original_shape(void);
extern "C" int patched_shape(void);

int main() {
    int executions = 0;
    for (int m : {1, 2, 4, 7, 16, 64}) for (int k : {1, 7, 64}) for (int n : {4, 8, 17, 64}) {
        for (int repeat = 0; repeat < 4; ++repeat) {
            std::vector<float> a(size_t(m)*k), b(size_t(k)*n), old(size_t(m)*n+32, 123.25f), changed;
            for (size_t i = 0; i < a.size(); ++i) a[i] = float(int((i*17 + repeat*11)%127)-63)/32;
            for (size_t i = 0; i < b.size(); ++i) b[i] = float(int((i*23 + repeat*7)%97)-48)/64;
            for (size_t i = 0; i < size_t(m)*n; ++i) old[i+16] = float(int((i*13+repeat)%53)-26)/16;
            if (repeat == 3) { a[0] = -0.f; b[0] = 0.f; uint32_t bits=1; std::memcpy(&a.back(), &bits, 4); }
            changed = old; auto before_a = a, before_b = b;
            original_gemm(old.data()+16, a.data(), b.data(), m, k, n);
            patched_gemm(changed.data()+16, a.data(), b.data(), m, k, n);
            assert(std::memcmp(old.data(), changed.data(), old.size()*sizeof(float)) == 0);
            assert(std::memcmp(a.data(), before_a.data(), a.size()*sizeof(float)) == 0);
            assert(std::memcmp(b.data(), before_b.data(), b.size()*sizeof(float)) == 0);
            for (int i=0;i<16;++i) assert(old[i]==123.25f && old[16+size_t(m)*n+i]==123.25f);
            ++executions;
        }
    }
    std::printf("{\"status\":\"FLASH_ARM_GEMM_PASS\",\"executions\":%d,\"original_shape\":%d,\"patched_shape\":%d,\"bitwise_equal\":true}\n",
                executions, original_shape(), patched_shape());
}
