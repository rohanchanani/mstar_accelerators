#pragma once
#include <cstddef>
#include <cstdint>

// Source-only physical metadata for a distinct narrow pipeline of the existing
// managed q8_0_dot_x4 operation. Other shapes keep its existing GPU pipeline.
namespace managed_q8_block_dot {
constexpr uint32_t workgroup=64, subgroup=16, rows=4, max_blocks=16;
constexpr uint32_t descriptors=5, shared_bytes=0;
struct Parameters {
    uint32_t ncols, stride_a, stride_b, stride_d;
    uint32_t batch_stride_a, batch_stride_b, batch_stride_d;
    uint32_t fusion_flags, base_work_group_y;
    uint32_t ne02, ne12, broadcast2, broadcast3;
};
static_assert(sizeof(Parameters)==52);
static_assert(offsetof(Parameters,broadcast3)==48);
inline bool physical_shape(uint64_t k, uint64_t m, uint64_t n) {
    if (!k || k%32 || k/32>max_blocks || !m || n!=1) return false;
    const uint64_t bytes_per_row=k/32*34;
    if (m>UINT32_MAX/bytes_per_row || m>UINT32_MAX/4) return false;
    return (m*bytes_per_row)%4==0;
}
inline bool parameters(const Parameters & p) {
    return physical_shape(p.ncols,p.stride_d,1) &&
        p.stride_a==p.ncols && p.stride_b==p.ncols &&
        p.batch_stride_a==0 && p.batch_stride_b==p.ncols &&
        p.batch_stride_d==p.stride_d && p.fusion_flags==0 &&
        p.base_work_group_y==0 && p.ne02==1 && p.ne12==1 &&
        p.broadcast2==1 && p.broadcast3==1;
}
} // namespace managed_q8_block_dot
