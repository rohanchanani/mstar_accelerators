// Provider-order mechanism only; no hardware/quantization qualification claim.
#include "../q8_provider.h"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool value) {
    if (!value) throw std::runtime_error("Q8 provider-order control failed");
}

int main() {
    using mstar_cpu_ggml::first_supported_q8_provider;
    auto name = [](const char * p) { return p; };
    std::vector<std::string> probed;
    auto accepts = [&](const char * p) { probed.emplace_back(p); return true; };
    const char * no_kai[] = {"CPU", "UNKNOWN_EXTRA", "CPU_REPACK", nullptr};
    check(std::string(first_supported_q8_provider(no_kai, name, accepts)) == "CPU_REPACK");
    check(probed == std::vector<std::string>{"CPU_REPACK"});
    const char * kai_first[] = {"CPU_KLEIDIAI", "CPU_REPACK", nullptr};
    probed.clear();
    check(std::string(first_supported_q8_provider(kai_first, name, accepts)) == "CPU_KLEIDIAI");
    check(probed == std::vector<std::string>{"CPU_KLEIDIAI"});
    const char * repack_first[] = {"CPU_REPACK", "CPU_KLEIDIAI", nullptr};
    probed.clear();
    check(std::string(first_supported_q8_provider(repack_first, name, accepts)) == "CPU_REPACK");
    check(probed == std::vector<std::string>{"CPU_REPACK"});
    probed.clear();
    auto reject_kai = [&](const char * p) {
        probed.emplace_back(p);
        return std::string(p) != "CPU_KLEIDIAI";
    };
    check(std::string(first_supported_q8_provider(kai_first, name, reject_kai)) == "CPU_REPACK");
    check(probed == std::vector<std::string>{"CPU_KLEIDIAI", "CPU_REPACK"});
    auto rejects = [](const char *) { return false; };
    check(first_supported_q8_provider(kai_first, name, rejects) == nullptr);
    check(first_supported_q8_provider(static_cast<const char * const *>(nullptr), name, accepts) == nullptr);
    const char * default_only[] = {"CPU", nullptr};
    check(first_supported_q8_provider(default_only, name, accepts) == nullptr);
    std::puts("Q8_PROVIDER_ORDER_MECHANISM_PASS hardware_and_quantization_not_tested");
}
