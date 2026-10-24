#pragma once

#include <ATen/ATen.h>
#include <mstar_pi/runtime.h>

namespace pybind11 { class module_; }

namespace mstar::pi::torch_binding {

// A checked entry into the one runtime owned by this Torch extension. Providers
// consume this binding; they do not construct another process registry or infer
// device ownership from Tensor.data_ptr().
struct Execution {
    std::shared_ptr<Runtime> runtime;
    StreamId stream;
};

Execution current_execution();
View tensor_view(const at::Tensor& tensor);

// Called by the extension's existing registration point. Importing the module
// only registers the API; provider device creation remains explicit preparation.
void register_ggml(pybind11::module_& module);

} // namespace mstar::pi::torch_binding
