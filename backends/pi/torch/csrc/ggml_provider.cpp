// Actual ggml Vulkan execution over the native runtime's canonical storage.
// This file is in the same extension as backend.cpp: it has no second allocator,
// stream registry, or lifetime owner for a Torch Storage.
#include "native_binding.h"

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-vulkan-managed.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

namespace py = pybind11;
namespace native = mstar::pi;

namespace mstar::pi::torch_binding {
namespace {

[[noreturn]] void provider_error(const char* operation, native::Runtime* outside = nullptr) {
  const bool fatal = ggml_vk_managed_error_is_fatal();
  const char* detail = ggml_vk_managed_last_error();
  if (fatal) {
    if (outside) outside->report_provider_failure(detail && *detail ? detail : operation);
    // Inside submit/retire/completion callbacks Runtime already owns its
    // ordering lock and translates this typed error into fatal containment.
    throw native::ProviderFatalError{};
  }
  throw native::Rejected(std::string(operation) + ": " +
                         (detail && *detail ? detail : "ggml provider failed"));
}

void check_status(int status, const char* operation, native::Runtime* outside = nullptr) {
  if (status != 0) provider_error(operation, outside);
}

struct OwnedFd {
  explicit OwnedFd(int value = -1) : value(value) {}
  ~OwnedFd() { if (value >= 0) ::close(value); }
  OwnedFd(const OwnedFd&) = delete;
  OwnedFd& operator=(const OwnedFd&) = delete;
  int release() noexcept { return std::exchange(value, -1); }
  int value;
};

// A buffer import aliases a native allocation. It does not own a new native BO
// or register a new StorageOwner. The native View is the lifetime anchor.
struct Import {
  native::View whole;
  ggml_backend_buffer_t buffer = nullptr;
  ~Import() { if (buffer) ggml_backend_buffer_free(buffer); }
};

struct Tail final : native::ProviderCompletion {
  ggml_vk_managed_tail_t value = nullptr;
  ~Tail() override {
    // The API refuses uncertain destruction. In that case the opaque object
    // keeps its actual VkDevice and physical receipt alive; never guess idle.
    if (value) (void)ggml_vk_managed_tail_destroy(value);
  }
  bool query() override {
    if (!value) throw native::FatalError("ggml completion has no tail");
    const int status = ggml_vk_managed_tail_query(value);
    if (status < 0) provider_error("ggml tail query");
    return status == 0;
  }
  void wait(std::chrono::nanoseconds duration) override {
    if (!value) throw native::FatalError("ggml completion has no tail");
    const auto timeout = duration.count() > 0 ? uint64_t(duration.count()) : uint64_t(0);
    if (ggml_vk_managed_tail_wait(value, timeout) < 0) provider_error("ggml tail wait");
  }
};

struct Slot {
  ggml_vk_managed_t managed = nullptr;
  ggml_context* metadata = nullptr;
  ggml_cgraph* graph = nullptr;
  ggml_tensor* w = nullptr;
  ggml_tensor* x = nullptr;
  ggml_tensor* y = nullptr;
  std::int64_t tokens = 0;
  bool busy = false;
  // Weak X/Y imports do not turn a persistent program into a tensor cache.
  std::map<std::uint64_t, std::weak_ptr<Import>> imports;
  std::shared_ptr<Import> weights;

  ~Slot() {
    if (metadata) ggml_free(metadata);
  }
};

struct SlotDeleter {
  void operator()(Slot* slot) const noexcept {
    // Busy Slots are held by their Submission; Native Pending retains that
    // Submission on uncertainty. Inactive managed destruction only frees
    // objects, with no fallible reset or wait. If that precondition is violated,
    // retain the entire existing Slot in place without allocating in a deleter.
    if (slot->managed && ggml_vk_managed_destroy(slot->managed) != 0) return;
    delete slot;
  }
};
using SlotPtr = std::unique_ptr<Slot, SlotDeleter>;

void clear_binding(ggml_tensor* tensor) noexcept {
  tensor->buffer = nullptr;
  tensor->data = nullptr;
  tensor->extra = nullptr;
}

void bind_tensor(ggml_tensor* tensor, const std::shared_ptr<Import>& imported,
                 const native::View& view) {
  const auto alignment = ggml_backend_buffer_get_alignment(imported->buffer);
  if (!alignment || view.offset % alignment != 0)
    throw native::Rejected("ggml tensor offset does not satisfy the selected Vulkan buffer alignment");
  const auto bytes = ggml_nbytes(tensor);
  if (bytes != view.bytes || view.offset > imported->whole.bytes ||
      bytes > imported->whole.bytes - view.offset)
    throw native::Rejected("ggml tensor shape/extent does not match its checked native view");
  const auto base = reinterpret_cast<std::uintptr_t>(ggml_backend_buffer_get_base(imported->buffer));
  if (!base) throw native::Rejected("ggml imported buffer has no binding address");
  if (view.offset > std::numeric_limits<std::uintptr_t>::max() - base)
    throw native::Rejected("ggml encoded binding address overflows");
  clear_binding(tensor);
  if (ggml_backend_tensor_alloc(imported->buffer, tensor,
                                reinterpret_cast<void*>(base + view.offset)) != GGML_STATUS_SUCCESS)
    throw native::Rejected("ggml tensor binding failed");
}

bool overlaps(const native::View& a, const native::View& b) {
  return a.allocation->identity() == b.allocation->identity() &&
         a.offset < b.offset + b.bytes && b.offset < a.offset + a.bytes;
}

struct Program {
  std::shared_ptr<native::Runtime> runtime;
  native::View weights;
  std::int64_t inputs = 0;
  std::int64_t outputs = 0;
  std::vector<std::int64_t> token_counts;
  std::size_t slots_per_shape = 0;
  std::size_t buffer_alignment = 0;
  mutable std::mutex mutex;
  std::vector<SlotPtr> slots;
  std::uint64_t submissions = 0;
  std::uint64_t retirements = 0;

  std::shared_ptr<Import> import(Slot& slot, const native::View& view) {
    // A leased slot is exclusive. No Program mutex is held across calls back
    // into Runtime, whose retirement path can in turn release another slot.
    std::erase_if(slot.imports, [](const auto& entry) { return entry.second.expired(); });
    const auto identity = view.allocation->identity();
    const auto found = slot.imports.find(identity);
    if (found != slot.imports.end()) {
      if (auto existing = found->second.lock()) return existing;
    }
    auto result = std::make_shared<Import>();
    result->whole = {view.allocation, 0, view.allocation->size()};
    OwnedFd fd(runtime->export_dma_buf(result->whole));
    result->buffer = ggml_vk_managed_import_dma_buf(
        slot.managed, fd.value, result->whole.bytes, view.allocation->backing_size());
    if (!result->buffer) provider_error("ggml DMA-BUF import", runtime.get());
    slot.imports[identity] = result;
    return result;
  }
};

struct Flight final : native::ProviderSubmission {
  std::shared_ptr<Program> program;
  Slot* slot;
  std::vector<std::shared_ptr<Import>> imports;
  std::shared_ptr<Tail> tail = std::make_shared<Tail>();
  OwnedFd unreturned_fd;
  bool accepted = false;
  bool retired = false;

  Flight(std::shared_ptr<Program> p, Slot* s) : program(std::move(p)), slot(s) {}

  ~Flight() override {
    // A postacceptance failure is quarantined by Runtime and never reaches
    // this destructor. Preaccept rejection may release a leased slot normally.
    if (accepted && !retired) std::terminate();
    clear_binding(slot->x);
    clear_binding(slot->y);
    imports.clear();
    std::lock_guard lock(program->mutex);
    slot->busy = false;
  }

  struct Acceptance {
    native::SubmissionAcceptance* native;
    Flight* flight;
  };

  static void may_submit(void* pointer) noexcept {
    auto& scope = *static_cast<Acceptance*>(pointer);
    scope.flight->accepted = true;
    scope.native->may_submit();
  }

  native::ProviderReceipt submit(int dependency_fd,
                                 native::SubmissionAcceptance& acceptance) override {
    Acceptance marker{&acceptance, this};
    const int status = ggml_vk_managed_submit(
        slot->managed, slot->graph, dependency_fd, may_submit, &marker,
        &tail->value, &unreturned_fd.value);
    if (status != 0) provider_error("ggml graph submission");
    if (!accepted || !tail->value || unreturned_fd.value < -1)
      throw native::FatalError("ggml returned an incomplete submission receipt");
    return {unreturned_fd.release(), tail};
  }

  void retire() override {
    // Runtime checked every prerequisite, provider tail and native health
    // before reaching this hook. An error poisons and retains this Flight.
    check_status(ggml_vk_managed_retire(slot->managed), "ggml graph retirement");
    retired = true;
    std::lock_guard lock(program->mutex);
    ++program->retirements;
  }
};

// Deliberately broken *receipt*, behind an explicitly opted-in test entry point.
// The real ggml work and its good receipt remain owned by this wrapper. Runtime
// must poison and retain them when the returned receipt is invalid. This does
// not manufacture a failed fence, reset the GPU, or alter the actual program.
struct MissingReceipt final : native::ProviderSubmission {
  explicit MissingReceipt(std::shared_ptr<Flight> flight) : flight(std::move(flight)) {}
  std::shared_ptr<Flight> flight;
  OwnedFd actual_fd;
  std::shared_ptr<native::ProviderCompletion> actual_completion;
  native::ProviderReceipt submit(int dependency_fd,
                                 native::SubmissionAcceptance& acceptance) override {
    auto actual = flight->submit(dependency_fd, acceptance);
    actual_fd.value = actual.sync_file;
    actual_completion = std::move(actual.completion);
    return {-2, {}};
  }
  void retire() override { flight->retire(); }
};

class PreparedLinear final {
public:
  PreparedLinear(const at::Tensor& weights, std::vector<std::int64_t> token_counts,
                 std::size_t slots_per_shape) {
    // Resolve the process-owned runtime before any ggml/Vulkan initialization.
    const auto execution = current_execution();
    if (weights.scalar_type() != at::kFloat || weights.dim() != 2 ||
        weights.size(0) <= 0 || weights.size(1) <= 0)
      throw native::Rejected("prepared linear requires nonempty float32 weights [outputs, inputs]");
    const auto weight_view = tensor_view(weights);
    if (token_counts.empty() || slots_per_shape == 0)
      throw native::Rejected("prepared linear requires token counts and at least one slot per shape");
    std::sort(token_counts.begin(), token_counts.end());
    if (token_counts.front() <= 0 ||
        std::adjacent_find(token_counts.begin(), token_counts.end()) != token_counts.end())
      throw native::Rejected("prepared token counts must be positive and unique");
    auto p = std::make_shared<Program>();
    p->runtime = execution.runtime;
    p->weights = weight_view;
    p->inputs = weights.size(1);
    p->outputs = weights.size(0);
    p->token_counts = std::move(token_counts);
    p->slots_per_shape = slots_per_shape;
    const auto device = p->runtime->device_identity();
    for (const auto tokens : p->token_counts) {
      if (tokens > std::numeric_limits<std::int64_t>::max() / std::max(p->inputs, p->outputs) / 4)
        throw native::Rejected("prepared linear tensor extent overflows");
      for (std::size_t index = 0; index < slots_per_shape; ++index) {
        SlotPtr slot(new Slot);
        slot->tokens = tokens;
        slot->managed = ggml_vk_managed_create(device.major, device.minor);
        if (!slot->managed) provider_error("ggml managed context creation", p->runtime.get());
        const std::size_t metadata_bytes = 8 * ggml_tensor_overhead() +
                                           ggml_graph_overhead_custom(8, false);
        slot->metadata = ggml_init({metadata_bytes, nullptr, true});
        if (!slot->metadata) throw native::Rejected("ggml metadata allocation failed");
        slot->w = ggml_new_tensor_2d(slot->metadata, GGML_TYPE_F32, p->inputs, p->outputs);
        slot->x = ggml_new_tensor_2d(slot->metadata, GGML_TYPE_F32, p->inputs, tokens);
        slot->y = ggml_mul_mat(slot->metadata, slot->w, slot->x);
        ggml_set_name(slot->w, "weights");
        ggml_set_name(slot->x, "input");
        ggml_set_name(slot->y, "output");
        // This snapshot supports source-precision override for src1 only.
        // src0 is F32 storage; the admitted path forbids its conversion.
        if (!ggml_prec_set_acc(slot->y, GGML_PREC_F32) ||
            !ggml_prec_set_src(slot->y, GGML_PREC_F32, 1))
          throw native::Rejected("ggml cannot express required float32 linear precision");
        slot->graph = ggml_new_graph_custom(slot->metadata, 8, false);
        ggml_build_forward_expand(slot->graph, slot->y);
        slot->weights = p->import(*slot, weight_view);
        const auto alignment = ggml_backend_buffer_get_alignment(slot->weights->buffer);
        if (!p->buffer_alignment) p->buffer_alignment = alignment;
        if (p->buffer_alignment != alignment)
          throw native::Rejected("managed ggml contexts disagree about buffer alignment");
        bind_tensor(slot->w, slot->weights, weight_view);
        // X/Y remain unbound. Pipeline preparation does not need disposable
        // full-size input/output allocations or execute a dummy graph.
        check_status(ggml_vk_managed_prepare(slot->managed, slot->graph),
                     "ggml graph preparation", p->runtime.get());
        p->slots.push_back(std::move(slot));
      }
    }
    program_ = std::move(p);
  }

  void submit(const at::Tensor& x, const at::Tensor& out) { submit_impl(x, out, false); }

  void test_missing_receipt(const at::Tensor& x, const at::Tensor& out,
                            const std::string& acknowledge) {
    if (acknowledge != "quarantine until supervised recovery")
      throw native::Rejected("missing-receipt test requires explicit quarantine/recovery acknowledgement");
    submit_impl(x, out, true);
  }

private:
  void submit_impl(const at::Tensor& x, const at::Tensor& out, bool missing_receipt) {
    const auto execution = current_execution();
    const auto p = program_;
    if (execution.runtime != p->runtime)
      throw native::Rejected("prepared linear belongs to another native context");
    if (x.scalar_type() != at::kFloat || out.scalar_type() != at::kFloat ||
        x.dim() != 2 || out.dim() != 2 || x.size(1) != p->inputs ||
        out.size(1) != p->outputs || out.size(0) != x.size(0))
      throw native::Rejected("linear expects float32 input [tokens, inputs] and output [tokens, outputs]");
    const auto input_view = tensor_view(x);
    const auto output_view = tensor_view(out);
    if (overlaps(output_view, input_view) || overlaps(output_view, p->weights))
      throw native::Rejected("linear output must not overlap input or weights");
    if (!std::binary_search(p->token_counts.begin(), p->token_counts.end(), x.size(0)))
      throw native::Rejected("linear token count was not prepared");

    // Retirement may call Flight::~Flight and take p->mutex. Never hold it
    // while entering Runtime. Lack of a free slot is explicit backpressure,
    // not an implicit host wait or an unsafe reset of in-flight descriptors.
    p->runtime->collect();
    Slot* chosen = nullptr;
    {
      std::lock_guard lock(p->mutex);
      for (auto& slot : p->slots) {
        if (slot->tokens == x.size(0) && !slot->busy) {
          chosen = slot.get();
          chosen->busy = true;
          break;
        }
      }
    }
    if (!chosen) throw native::Rejected("all prepared linear slots for this shape are in flight");

    // Arm a scope before any fallible import/binding. On preacceptance failure
    // its destructor returns the slot; Runtime owns it before the queue call.
    std::shared_ptr<Flight> flight;
    try {
      flight = std::make_shared<Flight>(p, chosen);
    } catch (...) {
      std::lock_guard lock(p->mutex);
      chosen->busy = false;
      throw;
    }
    auto input = p->import(*chosen, input_view);
    flight->imports.push_back(input);
    auto output = p->import(*chosen, output_view);
    flight->imports.push_back(output);
    bind_tensor(chosen->x, input, input_view);
    bind_tensor(chosen->y, output, output_view);
    std::shared_ptr<native::ProviderSubmission> submission = flight;
    if (missing_receipt) submission = std::make_shared<MissingReceipt>(flight);
    p->runtime->submit_provider(execution.stream, {p->weights, input_view, output_view}, submission);
    std::lock_guard lock(p->mutex);
    ++p->submissions;
  }

public:
  py::dict info() const {
    const auto p = program_;
    std::lock_guard lock(p->mutex);
    std::size_t active = 0;
    for (const auto& slot : p->slots) active += slot->busy;
    py::dict result;
    result["provider"] = "ggml-vulkan-managed";
    result["operation"] = "out = input @ weights.T";
    result["input_width"] = p->inputs;
    result["output_width"] = p->outputs;
    result["token_counts"] = p->token_counts;
    result["slots_per_shape"] = p->slots_per_shape;
    result["buffer_offset_alignment"] = p->buffer_alignment;
    result["active_slots"] = active;
    result["submissions"] = p->submissions;
    result["retirements"] = p->retirements;
    result["storage_dtype"] = "float32";
    result["source_precision"] = "float32";
    result["accumulation_precision"] = "float32";
    result["scratch_bytes_per_slot"] = 0;
    result["implicit_host_copies"] = false;
    return result;
  }

private:
  std::shared_ptr<Program> program_;
};

#include "graph_provider.inc"

} // namespace

void register_ggml(py::module_& module) {
  register_graph(module);
  py::class_<PreparedLinear, std::shared_ptr<PreparedLinear>>(module, "PreparedLinear")
      .def(py::init<const at::Tensor&, std::vector<std::int64_t>, std::size_t>(),
           py::arg("weights"), py::arg("token_counts"), py::arg("slots") = 2)
      .def("submit", &PreparedLinear::submit, py::arg("input"), py::arg("out"))
      .def("_test_submit_missing_receipt", &PreparedLinear::test_missing_receipt,
           py::arg("input"), py::arg("out"), py::kw_only(), py::arg("acknowledge"))
      .def("info", &PreparedLinear::info);
}

} // namespace mstar::pi::torch_binding
