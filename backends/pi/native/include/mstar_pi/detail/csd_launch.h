#pragma once

#include "mstar_pi/runtime.h"

namespace mstar::pi::detail {

inline bool supports_csd_device(std::uint32_t version, std::uint32_t revision) {
    // V3D 7.1.6 changed CFG4 from batch_count-1 to batch_count. The prepared
    // machine-code provider is initially scoped to that V3D 7.1 generation.
    return version == 71 && revision >= 6;
}

inline std::array<std::uint32_t, 5> encode_csd_launch(
    const CsdLaunch& launch, std::uint32_t version, std::uint32_t revision) {
    if (!supports_csd_device(version, revision))
        throw Rejected("CSD provider requires V3D 7.1 revision 6 or later");
    if (launch.workgroup_count != std::array<std::uint32_t, 3>{{1, 1, 1}} ||
        launch.workgroup_size != 16 || launch.workgroups_per_supergroup != 16 ||
        launch.batch_count != 1)
        throw Rejected("CSD provider currently supports only one 16-lane workgroup, 16 workgroups per supergroup, and one batch");

    // Register fields follow the supplied Mesa v3d_csd.h and v3dv_cmd_buffer.c.
    // Zero in the four-bit WGS_PER_SG field encodes sixteen workgroups.
    const auto batches_per_supergroup =
        (launch.workgroups_per_supergroup * launch.workgroup_size + 15) / 16;
    return {{launch.workgroup_count[0] << 16,
             launch.workgroup_count[1] << 16,
             launch.workgroup_count[2] << 16,
             ((batches_per_supergroup - 1) << 12) |
                 ((launch.workgroups_per_supergroup & 0xf) << 8) |
                 (launch.workgroup_size & 0xff),
             launch.batch_count}};
}

} // namespace mstar::pi::detail
