#pragma once

#include <cstring>

namespace mstar_cpu_ggml {

// Preserve the upstream registry order. This component explicitly admits only
// the two upstream Q8 weight representations, never a default-CPU fallback.
template<class Provider, class Name, class Supports>
Provider first_supported_q8_provider(const Provider * choices, Name name, Supports supports) {
    for (; choices && *choices; ++choices) {
        const char * value = name(*choices);
        if (value && (std::strcmp(value, "CPU_KLEIDIAI") == 0 || std::strcmp(value, "CPU_REPACK") == 0)
                && supports(*choices)) {
            return *choices;
        }
    }
    return {};
}

}  // namespace mstar_cpu_ggml
