// Torch 2.9.1 adapter. Native handles stay behind mstar_pi/runtime.h.
#include <mstar_pi/runtime.h>
#include "native_binding.h"

#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/ops/alias_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <c10/core/Allocator.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/library.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <unistd.h>

namespace py = pybind11;
namespace native = mstar::pi;

namespace {

constexpr auto kDevice = c10::DeviceType::PrivateUse1;
thread_local native::StreamId current_stream = 0;

struct StorageOwner;

struct ProcessState {
  std::mutex mutex;
  std::shared_ptr<native::Runtime> runtime;
  std::atomic<pid_t> pid{0};
  bool ever_initialized = false;
  bool closed = false;
  std::map<std::uintptr_t, StorageOwner*> storage;
  StorageOwner* retirement_holds = nullptr;
  std::size_t retained_storage_owners = 0;
  std::size_t retirement_failures = 0;
};

// Torch's guard registry itself intentionally survives static destruction.
// This registry follows that lifetime. Explicit shutdown performs the drain;
// allocation deleters remain callable during Python/module destruction.
ProcessState& process() {
  static auto* state = new ProcessState;
  return *state;
}

bool bad_fork() noexcept {
  const auto pid = process().pid.load(std::memory_order_acquire);
  return pid != 0 && pid != getpid();
}

void check_process() {
  TORCH_CHECK(!bad_fork(), "mstar_pi native context inherited across fork; use spawn");
}

std::shared_ptr<native::Runtime> runtime(bool require_admission = true) {
  check_process();
  auto& state = process();
  std::lock_guard lock(state.mutex);
  TORCH_CHECK(state.runtime && (!require_admission || !state.closed),
              "mstar_pi requires an initialized context with open admission for new work");
  return state.runtime;
}

void check_device(c10::Device device) {
  TORCH_CHECK(device.type() == kDevice && (device.index() == 0 || device.index() == -1),
              "mstar_pi exposes only device 0");
}

void check_stream(const c10::Stream& stream) {
  check_device(stream.device());
  TORCH_CHECK(stream.id() >= 0, "invalid Pi stream ID");
}

void check_dtype(at::ScalarType dtype) {
  TORCH_CHECK(dtype == at::kFloat || dtype == at::kInt || dtype == at::kLong,
              "mstar_pi first slice supports float32, int32 and int64 storage only");
}

void check_tensor(const at::Tensor& tensor) {
  check_dtype(tensor.scalar_type());
  TORCH_CHECK(tensor.layout() == at::kStrided && tensor.is_contiguous(),
              "mstar_pi first slice requires contiguous tensor operands");
  TORCH_CHECK(!tensor.requires_grad(), "mstar_pi first slice is inference-only");
  TORCH_CHECK(!tensor.is_conj() && !tensor.is_neg(),
              "mstar_pi does not implement lazy conjugate/negative views");
}

struct StorageOwner {
  std::shared_ptr<native::Runtime> runtime;
  std::shared_ptr<native::Allocation> allocation;
  std::size_t logical_size;
  bool pinned;
  std::mutex mutex;
  std::set<native::StreamId> recorded_streams;
  StorageOwner* retirement_next = nullptr;
};

void delete_storage(void* opaque) noexcept {
  auto* owner = static_cast<StorageOwner*>(opaque);
  if (!owner) return;
  // A child created by fork must not touch inherited driver state or locks.
  // Deliberately retain it; the inherited context is unusable in that process.
  if (bad_fork()) return;
  try {
    // record_stream protects uses issued after record_stream but before the
    // last storage reference dies: snapshot each recorded stream here.
    std::lock_guard lock(owner->mutex);
    for (auto stream : owner->recorded_streams)
      owner->runtime->retain_until(stream, owner->allocation);
  } catch (...) {
    // Failure is not proof of quiescence. An intrusive list needs no new
    // allocation on this failure path. Stop new work, expose the retention,
    // and release it only after successful device-wide shutdown/drain.
    auto& state = process();
    std::lock_guard lock(state.mutex);
    state.storage.erase(reinterpret_cast<std::uintptr_t>(owner->allocation->mapped()));
    owner->retirement_next = state.retirement_holds;
    state.retirement_holds = owner;
    ++state.retained_storage_owners;
    ++state.retirement_failures;
    state.closed = true;
    return;
  }
  {
    auto& state = process();
    std::lock_guard lock(state.mutex);
    // Keep the owner visible through retirement so shutdown cannot pass its
    // live-storage check while this deleter still needs a stream snapshot.
    state.storage.erase(reinterpret_cast<std::uintptr_t>(owner->allocation->mapped()));
  }
  delete owner;
}

StorageOwner* owned_storage(const at::Tensor& tensor) {
  const auto& ptr = tensor.storage().data_ptr();
  if (ptr.get_deleter() != &delete_storage) return nullptr;
  return static_cast<StorageOwner*>(ptr.get_context());
}

native::View view(const at::Tensor& tensor) {
  check_tensor(tensor);
  auto* owner = owned_storage(tensor);
  TORCH_CHECK(owner, "tensor has no mstar_pi-owned backing allocation");
  TORCH_CHECK(owner->runtime == runtime(), "tensor belongs to another Pi context");
  TORCH_CHECK(tensor.storage_offset() >= 0, "negative storage offset");
  TORCH_CHECK(static_cast<std::uint64_t>(tensor.storage_offset()) <=
                  std::numeric_limits<std::size_t>::max() / tensor.element_size(),
              "tensor storage offset overflows its native byte address");
  auto offset = static_cast<std::size_t>(tensor.storage_offset()) * tensor.element_size();
  auto bytes = static_cast<std::size_t>(tensor.nbytes());
  TORCH_CHECK(offset <= owner->logical_size && bytes <= owner->logical_size - offset,
              "tensor view exceeds its logical allocation extent");
  return {owner->allocation, offset, bytes};
}

class PiAllocator final : public c10::Allocator {
 public:
  explicit PiAllocator(bool pinned) : pinned_(pinned) {}

  c10::DataPtr allocate(std::size_t bytes) override {
    auto rt = runtime();
    auto allocation = rt->allocate(bytes);
    auto owner = std::make_unique<StorageOwner>();
    owner->runtime = rt;
    owner->allocation = std::move(allocation);
    owner->logical_size = bytes;
    owner->pinned = pinned_;
    void* mapped = owner->allocation->mapped();
    {
      auto& state = process();
      std::lock_guard lock(state.mutex);
      TORCH_CHECK(!state.closed && state.runtime == rt,
                  "Pi context closed while allocating tensor storage");
      TORCH_CHECK(state.storage.emplace(reinterpret_cast<std::uintptr_t>(mapped), owner.get()).second,
                  "duplicate native allocation mapping");
    }
    // PrivateUse1 data pointers are opaque to consumers. Native launch uses
    // the Allocation/offset metadata above, never this CPU mapping as a GPU VA.
    return {mapped, owner.release(), &delete_storage,
            pinned_ ? c10::Device(c10::kCPU) : c10::Device(kDevice, 0)};
  }

  void copy_data(void*, const void*, std::size_t) const override {
    TORCH_CHECK(false, "implicit allocator storage resize/copy is unsupported; use Tensor.copy_");
  }

 private:
  bool pinned_;
};

PiAllocator device_allocator(false);
PiAllocator pinned_allocator(true);

struct EventRecord {
  std::shared_ptr<native::Runtime> runtime;
  native::Event event;
};

class Guard final : public c10::impl::DeviceGuardImplInterface {
 public:
  c10::DeviceType type() const override { return kDevice; }
  c10::Device getDevice() const override { check_process(); return {kDevice, 0}; }
  void setDevice(c10::Device device) const override { check_device(device); (void)runtime(); }
  void uncheckedSetDevice(c10::Device) const noexcept override {}
  c10::Device exchangeDevice(c10::Device device) const override { setDevice(device); return {kDevice, 0}; }

  c10::Stream getStream(c10::Device device) const override {
    check_device(device); (void)runtime();
    return c10::Stream(c10::Stream::UNSAFE, {kDevice, 0}, static_cast<c10::StreamId>(current_stream));
  }
  c10::Stream getDefaultStream(c10::Device device) const override {
    check_device(device); (void)runtime();
    return c10::Stream(c10::Stream::DEFAULT, {kDevice, 0});
  }
  c10::Stream getNewStream(c10::Device device, int priority) const override {
    check_device(device);
    TORCH_CHECK(priority == 0, "mstar_pi has no stream-priority capability");
    const auto id = runtime()->new_stream();
    TORCH_CHECK(id <= static_cast<std::uint64_t>(std::numeric_limits<c10::StreamId>::max()),
                "Pi stream ID overflow");
    return c10::Stream(c10::Stream::UNSAFE, {kDevice, 0}, static_cast<c10::StreamId>(id));
  }
  c10::Stream getStreamFromGlobalPool(c10::Device device, bool high) const override {
    TORCH_CHECK(!high, "mstar_pi has no high-priority stream capability");
    return getNewStream(device, 0);
  }
  c10::Stream exchangeStream(c10::Stream stream) const override {
    check_stream(stream);
    // Validate the ID with the native stream registry, without waiting.
    (void)runtime()->record(static_cast<native::StreamId>(stream.id()));
    auto previous = getStream(stream.device());
    current_stream = static_cast<native::StreamId>(stream.id());
    return previous;
  }
  c10::DeviceIndex deviceCount() const noexcept override {
    if (bad_fork()) return 0;
    try {
      auto& state = process();
      std::lock_guard lock(state.mutex);
      if (!state.ever_initialized) return native::probe_device() ? 1 : 0;
      return state.runtime && !state.closed && !state.runtime->poisoned() ? 1 : 0;
    } catch (...) { return 0; }
  }

  void record(void** target, const c10::Stream& stream, c10::DeviceIndex index,
              c10::EventFlag flag) const override {
    check_stream(stream);
    TORCH_CHECK(index == -1 || index == 0, "invalid event device index");
    TORCH_CHECK(flag == c10::EventFlag::PYTORCH_DEFAULT,
                "mstar_pi timing events are unsupported");
    auto rt = runtime();
    EventRecord next{rt, rt->record(static_cast<native::StreamId>(stream.id()))};
    if (*target) *static_cast<EventRecord*>(*target) = std::move(next);
    else *target = new EventRecord(std::move(next));
  }
  void destroyEvent(void* event, c10::DeviceIndex) const noexcept override {
    if (!bad_fork()) delete static_cast<EventRecord*>(event);
  }
  void block(void* event, const c10::Stream& stream) const override {
    check_stream(stream);
    if (!event) return;
    const auto& point = *static_cast<EventRecord*>(event);
    auto rt = runtime();
    TORCH_CHECK(point.runtime == rt, "event belongs to another Pi context");
    rt->wait_event(static_cast<native::StreamId>(stream.id()), point.event);
  }
  bool queryEvent(void* event) const override {
    check_process();
    if (!event) return true;
    const auto& point = *static_cast<EventRecord*>(event);
    return point.runtime->query(point.event);
  }
  void synchronizeEvent(void* event) const override {
    check_process();
    if (!event) return;
    const auto& point = *static_cast<EventRecord*>(event);
    point.runtime->synchronize(point.event);
  }
  bool queryStream(const c10::Stream& stream) const override {
    check_stream(stream);
    return runtime()->query_stream(static_cast<native::StreamId>(stream.id()));
  }
  void synchronizeStream(const c10::Stream& stream) const override {
    check_stream(stream);
    runtime()->synchronize_stream(static_cast<native::StreamId>(stream.id()));
  }
  void synchronizeDevice(c10::DeviceIndex index) const override {
    TORCH_CHECK(index == -1 || index == 0, "invalid Pi device index");
    runtime(false)->synchronize_device();
  }
  void recordDataPtrOnStream(const c10::DataPtr& ptr, const c10::Stream& stream) const override {
    check_stream(stream);
    TORCH_CHECK(ptr.get_deleter() == &delete_storage, "unowned DataPtr passed to Pi record_stream");
    auto* owner = static_cast<StorageOwner*>(ptr.get_context());
    TORCH_CHECK(owner->runtime == runtime(), "DataPtr belongs to another Pi context");
    (void)owner->runtime->record(static_cast<native::StreamId>(stream.id()));
    std::lock_guard lock(owner->mutex);
    owner->recorded_streams.insert(static_cast<native::StreamId>(stream.id()));
  }
};

class Hooks final : public at::PrivateUse1HooksInterface {
 public:
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return Guard().deviceCount() != 0; }
  bool hasPrimaryContext(c10::DeviceIndex index) const override {
    if (bad_fork() || (index != -1 && index != 0)) return false;
    auto& state = process();
    std::lock_guard lock(state.mutex);
    return bool(state.runtime);
  }
  void init() const override { (void)runtime(); }
  at::Allocator* getPinnedMemoryAllocator() const override { (void)runtime(); return &pinned_allocator; }
  bool isPinnedPtr(const void* ptr) const override {
    if (!ptr || bad_fork()) return false;
    auto& state = process();
    std::lock_guard lock(state.mutex);
    auto address = reinterpret_cast<std::uintptr_t>(ptr);
    auto upper = state.storage.upper_bound(address);
    if (upper == state.storage.begin()) return false;
    const auto& [base, owner] = *std::prev(upper);
    return owner->pinned && address - base < owner->logical_size;
  }
  at::Device getDeviceFromPtr(void* ptr) const override {
    TORCH_CHECK(ptr, "null pointer has no Pi device identity");
    check_process();
    auto& state = process();
    std::lock_guard lock(state.mutex);
    auto address = reinterpret_cast<std::uintptr_t>(ptr);
    auto upper = state.storage.upper_bound(address);
    TORCH_CHECK(upper != state.storage.begin(), "pointer is not owned by Pi backend");
    const auto& [base, owner] = *std::prev(upper);
    TORCH_CHECK(!owner->pinned && address - base < owner->logical_size,
                "pointer is not a Pi device-storage address");
    return {kDevice, 0};
  }
};

Hooks hooks;

std::vector<int64_t> concrete(c10::SymIntArrayRef values) {
  std::vector<int64_t> result;
  result.reserve(values.size());
  for (const auto& value : values) result.push_back(value.expect_int());
  return result;
}

at::Tensor pi_as_strided(const at::Tensor& tensor, c10::SymIntArrayRef sizes,
                         c10::SymIntArrayRef strides, std::optional<c10::SymInt> offset) {
  return at::native::as_strided_tensorimpl(tensor, concrete(sizes), concrete(strides),
      offset ? std::optional<int64_t>(offset->expect_int()) : std::nullopt);
}

at::Tensor pi_view(const at::Tensor& tensor, c10::SymIntArrayRef sizes) {
  return at::native::view(tensor, concrete(sizes));
}

at::Tensor pi_reshape_alias(const at::Tensor& tensor, c10::SymIntArrayRef sizes,
                           c10::SymIntArrayRef strides) {
  return at::native::_reshape_alias(tensor, concrete(sizes), concrete(strides));
}

at::Tensor empty(c10::SymIntArrayRef size, std::optional<at::ScalarType> dtype,
                 std::optional<at::Layout> layout, std::optional<c10::Device> device,
                 std::optional<bool> pin_memory, std::optional<at::MemoryFormat> format) {
  check_device(device.value_or(c10::Device(kDevice, 0)));
  TORCH_CHECK(!layout || *layout == at::kStrided, "Pi supports strided storage only");
  TORCH_CHECK(!pin_memory.value_or(false), "pin_memory applies to CPU staging, not Pi tensors");
  TORCH_CHECK(!format || *format == at::MemoryFormat::Contiguous,
              "Pi empty supports contiguous memory format only");
  auto scalar = dtype.value_or(c10::typeMetaToScalarType(c10::get_default_dtype()));
  check_dtype(scalar);
  auto sizes = concrete(size);
  return at::detail::empty_generic(sizes, &device_allocator,
                                  c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), scalar, format);
}

at::Tensor empty_strided(c10::SymIntArrayRef size, c10::SymIntArrayRef stride,
                         std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
                         std::optional<c10::Device> device, std::optional<bool> pin_memory) {
  check_device(device.value_or(c10::Device(kDevice, 0)));
  TORCH_CHECK(!layout || *layout == at::kStrided, "Pi supports strided storage only");
  TORCH_CHECK(!pin_memory.value_or(false), "pin_memory applies to CPU staging, not Pi tensors");
  auto scalar = dtype.value_or(c10::typeMetaToScalarType(c10::get_default_dtype()));
  check_dtype(scalar);
  auto sizes = concrete(size), strides = concrete(stride);
  for (auto s : strides) TORCH_CHECK(s >= 0, "negative Pi strides are unsupported");
  return at::detail::empty_strided_generic(sizes, strides, &device_allocator,
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), scalar);
}

at::Tensor& copy(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
  check_tensor(dst); check_tensor(src);
  TORCH_CHECK(dst.sizes() == src.sizes(), "Pi copy requires equal shapes (no implicit broadcast)");
  TORCH_CHECK(dst.scalar_type() == src.scalar_type(), "Pi copy requires equal dtypes (no conversion)");
  const bool dst_pi = dst.device().type() == kDevice;
  const bool src_pi = src.device().type() == kDevice;
  TORCH_CHECK(dst_pi || src_pi, "Pi copy requires at least one Pi-device tensor");
  TORCH_CHECK((dst_pi || dst.device().is_cpu()) && (src_pi || src.device().is_cpu()),
              "Pi cross-accelerator copy requires an explicit transfer provider");
  if (dst_pi) check_device(dst.device());
  if (src_pi) check_device(src.device());
  auto rt = runtime();
  const auto bytes = static_cast<std::size_t>(dst.nbytes());
  if (!bytes) return dst;
  if (dst_pi && src_pi) {
    rt->copy(current_stream, view(dst), view(src));
  } else if (dst_pi) {
    if (auto* owner = owned_storage(src); owner && owner->pinned) {
      // A pinned source may itself have been written asynchronously. Avoid a
      // host read here: native copy consumes the stream's dependency prefix.
      rt->copy(current_stream, view(dst), view(src));
    } else {
      auto staging = rt->allocate(bytes);
      std::memcpy(staging->mapped(), src.const_data_ptr(), bytes);
      rt->copy(current_stream, view(dst), {staging, 0, bytes});
    }
    if (!non_blocking) rt->synchronize_stream(current_stream);
  } else {
    if (auto* owner = owned_storage(dst); owner && owner->pinned) {
      rt->copy(current_stream, view(dst), view(src));
      if (!non_blocking) rt->synchronize_stream(current_stream);
    } else {
      // A pageable destination has no backend-owned async CPU lifetime. The
      // CPU-read boundary waits, then performs the host copy. No pending writes
      // escape in an ordinary CPU tensor.
      auto staging = rt->allocate(bytes);
      rt->copy(current_stream, {staging, 0, bytes}, view(src));
      rt->synchronize_stream(current_stream);
      std::memcpy(dst.mutable_data_ptr(), staging->mapped(), bytes);
    }
  }
  return dst;
}

at::Tensor copy_from(const at::Tensor& src, const at::Tensor& dst, bool non_blocking) {
  auto result = dst;
  copy(result, src, non_blocking);
  return result;
}

at::Tensor pi_clone(const at::Tensor& src, std::optional<at::MemoryFormat> format) {
  check_tensor(src);
  TORCH_CHECK(!format || *format == at::MemoryFormat::Preserve || *format == at::MemoryFormat::Contiguous,
              "Pi clone supports contiguous/preserve format only");
  auto dst = at::empty(src.sizes(), src.options().memory_format(at::MemoryFormat::Contiguous));
  copy(dst, src, true);
  return dst;
}

at::Tensor& fill(at::Tensor& dst, const at::Scalar& value) {
  check_tensor(dst); check_device(dst.device());
  std::uint32_t bits;
  if (dst.scalar_type() == at::kFloat) {
    const float scalar = value.toFloat();
    std::memcpy(&bits, &scalar, sizeof(bits));
  } else if (dst.scalar_type() == at::kInt) {
    const auto scalar = value.toInt();
    std::memcpy(&bits, &scalar, sizeof(bits));
  } else {
    const auto scalar = static_cast<std::uint64_t>(value.toLong());
    bits = static_cast<std::uint32_t>(scalar);
    TORCH_CHECK(bits == static_cast<std::uint32_t>(scalar >> 32),
                "Pi first-slice int64 fill requires a repeated 32-bit pattern (including 0 and -1)");
  }
  runtime()->fill32(current_stream, view(dst), bits);
  return dst;
}

at::Tensor& zero(at::Tensor& dst) { return fill(dst, 0); }

at::Scalar local_scalar(const at::Tensor& src) {
  TORCH_CHECK(src.numel() == 1, "item requires a one-element tensor");
  auto host = at::empty(src.sizes(), src.options().device(at::kCPU));
  copy(host, src, false);
  if (src.scalar_type() == at::kFloat) return *host.const_data_ptr<float>();
  if (src.scalar_type() == at::kInt) return *host.const_data_ptr<int32_t>();
  return *host.const_data_ptr<int64_t>();
}

void record_stream(at::Tensor& tensor, c10::Stream stream) {
  check_device(tensor.device());
  Guard().recordDataPtrOnStream(tensor.storage().data_ptr(), stream);
}

struct BufferAddress {
  std::shared_ptr<native::Runtime> runtime;
  native::View view;
  std::size_t offset;
};

BufferAddress address(const at::Tensor& tensor, int64_t byte_offset) {
  auto bound = view(tensor);
  TORCH_CHECK(byte_offset >= 0 && static_cast<std::size_t>(byte_offset) < bound.bytes,
              "address offset must identify a byte within the logical tensor view");
  TORCH_CHECK(byte_offset % 4 == 0, "native CSD address operand must be four-byte aligned");
  return {runtime(), std::move(bound), static_cast<std::size_t>(byte_offset)};
}

struct PreparedCsd {
  std::shared_ptr<native::Runtime> runtime;
  std::shared_ptr<native::PreparedCsd> program;

  void submit(const py::sequence& uniform_args, const std::vector<at::Tensor>& uses) {
    TORCH_CHECK(runtime == ::runtime(), "native program belongs to another Pi context");
    std::vector<uint32_t> uniforms;
    std::vector<native::View> borrowed;
    for (const auto& value : uniform_args) {
      if (py::isinstance<BufferAddress>(value)) {
        const auto address = py::cast<BufferAddress>(value);
        TORCH_CHECK(address.runtime == runtime, "native address belongs to another Pi context");
        uniforms.push_back(address.view.allocation->gpu_address(address.view.offset + address.offset));
        borrowed.push_back(address.view);
      } else {
        const auto immediate = py::cast<std::uint64_t>(value);
        TORCH_CHECK(immediate <= UINT32_MAX, "CSD uniform must fit in uint32");
        uniforms.push_back(static_cast<uint32_t>(immediate));
      }
    }
    for (const auto& tensor : uses) borrowed.push_back(view(tensor));
    py::gil_scoped_release release;
    runtime->submit_csd(current_stream, program, uniforms, borrowed);
  }
};

void initialize(const std::string& shader, const std::string& node,
                int64_t timeout_ms, bool validation) {
  check_process();
  TORCH_CHECK(timeout_ms > 0, "positive wait timeout required");
  auto& state = process();
  std::lock_guard lock(state.mutex);
  TORCH_CHECK(!state.ever_initialized, "initialize once per process; use a new spawned worker for a new context");
  native::Options options;
  options.copy_shader = shader;
  options.render_node = node;
  options.wait_timeout = std::chrono::milliseconds(timeout_ms);
  options.validation = validation;
  state.pid.store(getpid(), std::memory_order_release);
  auto rt = native::Runtime::open(options);
  state.runtime = std::move(rt);
  state.pid.store(getpid(), std::memory_order_release);
  state.ever_initialized = true;
  current_stream = 0;
}

void shutdown() {
  check_process();
  auto& state = process();
  std::shared_ptr<native::Runtime> rt;
  {
    std::lock_guard lock(state.mutex);
    TORCH_CHECK(state.runtime, "Pi context is not initialized");
    TORCH_CHECK(state.storage.empty(), "release Pi and pinned-CPU tensors before shutdown");
    rt = state.runtime;
    state.closed = true;
  }
  rt->close();
  // Existing events and native program/address bindings retain their own
  // context ownership. On a successful close the registry itself need not.
  // If close throws, leave it present so health() preserves the fatal status.
  StorageOwner* released;
  {
    std::lock_guard lock(state.mutex);
    state.runtime.reset();
    released = state.retirement_holds;
    state.retirement_holds = nullptr;
    state.retained_storage_owners = 0;
  }
  while (released) {
    auto* next = released->retirement_next;
    delete released;
    released = next;
  }
}

py::dict health() {
  py::dict result;
  if (bad_fork()) {
    result["initialized"] = false;
    result["poisoned"] = true;
    result["reason"] = "native context inherited across fork";
    return result;
  }
  auto& state = process();
  std::lock_guard lock(state.mutex);
  result["initialized"] = bool(state.runtime);
  result["closed"] = state.closed;
  result["poisoned"] = state.runtime && state.runtime->poisoned();
  result["reason"] = state.runtime ? state.runtime->failure_reason() : std::string();
  result["live_storages"] = state.storage.size();
  result["retained_storage_owners"] = state.retained_storage_owners;
  result["retirement_failures"] = state.retirement_failures;
  result["retirement_reason"] = state.retirement_failures
      ? "Storage-owner retirement failed; admission stopped and owner retained until drain"
      : "";
  return result;
}

} // namespace

namespace mstar::pi::torch_binding {

Execution current_execution() {
  return {::runtime(), ::current_stream};
}

View tensor_view(const at::Tensor& tensor) {
  ::check_device(tensor.device());
  return ::view(tensor);
}

} // namespace mstar::pi::torch_binding

void register_kernels() {
  // Delay registration until the module has established exclusive ownership
  // of PrivateUse1. A static registration would clobber another backend before
  // the module's conflict check could reject the import. These registrations
  // have the same process lifetime as Torch's non-owning guard registry.
  auto library = std::make_unique<torch::Library>(torch::Library::IMPL, "aten",
      c10::DispatchKey::PrivateUse1, __FILE__, __LINE__);
  library->impl("empty.memory_format", TORCH_FN(empty));
  library->impl("empty_strided", TORCH_FN(empty_strided));
  library->impl("copy_", TORCH_FN(copy));
  library->impl("_copy_from", TORCH_FN(copy_from));
  library->impl("clone", TORCH_FN(pi_clone));
  library->impl("fill_.Scalar", TORCH_FN(fill));
  library->impl("zero_", TORCH_FN(zero));
  library->impl("_local_scalar_dense", TORCH_FN(local_scalar));
  library->impl("record_stream", TORCH_FN(record_stream));
  // These native helpers only manipulate TensorImpl metadata; they do not
  // execute CPU math or dereference the device data pointer.
  library->impl("as_strided", TORCH_FN(pi_as_strided));
  library->impl("view", TORCH_FN(pi_view));
  library->impl("_reshape_alias", TORCH_FN(pi_reshape_alias));
  auto autograd = std::make_unique<torch::Library>(torch::Library::IMPL, "_",
      c10::DispatchKey::AutogradPrivateUse1, __FILE__, __LINE__);
  autograd->fallback(torch::CppFunction::makeFallthrough());
  // Keep successful registrations for the process lifetime. If a schema check
  // above throws during import, unique_ptr rolls every prior kernel back.
  (void)library.release();
  (void)autograd.release();
}

PYBIND11_MODULE(_C, module) {
  TORCH_CHECK(!at::isPrivateUse1HooksRegistered() && !c10::impl::hasDeviceGuardImpl(kDevice),
              "another extension already owns PrivateUse1 hooks");
  register_kernels();
  c10::impl::DeviceGuardImplRegistrar registration(kDevice, new Guard);
  at::RegisterPrivateUse1HooksInterface(&hooks);
  c10::SetAllocator(kDevice, &device_allocator);
  py::register_exception<native::FatalError>(module, "FatalError");
  py::register_exception<native::Rejected>(module, "Rejected");
  module.def("initialize", &initialize, py::call_guard<py::gil_scoped_release>());
  module.def("shutdown", &shutdown, py::call_guard<py::gil_scoped_release>());
  module.def("is_available", [] { return Guard().deviceCount() != 0; });
  module.def("probe", &native::probe_device, py::arg("render_node") = "/dev/dri/renderD128");
  module.def("is_initialized", [] {
    if (bad_fork()) return false;
    auto& state = process(); std::lock_guard lock(state.mutex);
    return bool(state.runtime);
  });
  module.def("is_in_bad_fork", &bad_fork);
  module.def("health", &health);
  module.def("synchronize", [] { runtime(false)->synchronize_device(); }, py::call_guard<py::gil_scoped_release>());
  module.def("collect", [] { runtime()->collect(); });
  py::class_<BufferAddress>(module, "BufferAddress");
  module.def("address", &address, py::arg("tensor"), py::arg("byte_offset") = 0);
  py::class_<PreparedCsd>(module, "PreparedCsd")
      .def("submit", &PreparedCsd::submit, py::arg("uniforms"), py::arg("uses") = std::vector<at::Tensor>{});
  module.def("prepare_csd", [](const std::vector<std::uint64_t>& words) {
    auto rt = runtime();
    return PreparedCsd{rt, rt->prepare_csd(words)};
  });
  mstar::pi::torch_binding::register_ggml(module);
}
