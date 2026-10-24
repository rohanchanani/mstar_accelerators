#include "mstar_pi/runtime.h"
#include "mstar_pi/detail/csd_launch.h"

#include <iostream>
#include <stdexcept>

static void require(bool value) {
    if (!value) throw std::runtime_error("native preflight check failed");
}

int main() {
    require(!mstar::pi::probe_device("/this/path/does/not/exist"));
    require(!mstar::pi::probe_device("/dev/null"));
    int rejected = 0;
    try { mstar::pi::Runtime::open({}); }
    catch (const mstar::pi::Rejected&) { ++rejected; }
    mstar::pi::Options options;
    options.copy_shader = "unused-before-native-admission.spv";
    options.wait_timeout = std::chrono::milliseconds(0);
    try { mstar::pi::Runtime::open(options); }
    catch (const mstar::pi::Rejected&) { ++rejected; }
    options.wait_timeout = std::chrono::milliseconds(1);
    options.render_node = "/dev/null";
    try { mstar::pi::Runtime::open(options); }
    catch (const mstar::pi::Rejected&) { ++rejected; }
    require(rejected == 3);
    const auto cfg = mstar::pi::detail::encode_csd_launch({}, 71, 6);
    require(cfg[0] == 1u << 16 && cfg[1] == 1u << 16 && cfg[2] == 1u << 16);
    require(cfg[3] == 0xf010 && cfg[4] == 1);
    mstar::pi::CsdLaunch unsupported;
    unsupported.workgroup_count[0] = 2;
    try { mstar::pi::detail::encode_csd_launch(unsupported, 71, 6); }
    catch (const mstar::pi::Rejected&) { ++rejected; }
    try { mstar::pi::detail::encode_csd_launch({}, 71, 5); }
    catch (const mstar::pi::Rejected&) { ++rejected; }
    try { mstar::pi::detail::encode_csd_launch({}, 42, 1); }
    catch (const mstar::pi::Rejected&) { ++rejected; }
    require(rejected == 6);
    std::cout << "PASS: no-device discovery, native-contract preflight, and physical launch rejection\n";
}
