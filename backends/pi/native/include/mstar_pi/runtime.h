#pragma once

#include "mstar_pi/provider.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mstar::pi {

using StreamId = std::uint64_t;
namespace detail { struct State; struct AllocationState; struct EventState; }

// Rejection is before native acceptance. FatalError forbids continued use of
// this context; an error or timeout alone does not establish safe retirement.
class Rejected : public std::runtime_error { using std::runtime_error::runtime_error; };
class FatalError : public std::runtime_error { using std::runtime_error::runtime_error; };

struct Options {
    std::string render_node = "/dev/dri/renderD128";
    std::string copy_shader;  // SPIR-V compiled from shaders/transfer.comp.
    std::chrono::milliseconds wait_timeout{30000};
    bool validation = false;
};

// Discovery does not construct Vulkan objects, upload programs, or submit
// work. Runtime::open still validates the complete selected provider surface.
bool probe_device(const std::string& render_node = "/dev/dri/renderD128") noexcept;

class Allocation final {
public:
    ~Allocation();
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
    std::size_t size() const noexcept;
    std::size_t backing_size() const noexcept;
    void* mapped() const noexcept;
    std::uint32_t gpu_address(std::size_t offset = 0) const;
    std::uint64_t identity() const noexcept;
private:
    explicit Allocation(std::shared_ptr<detail::AllocationState>);
    std::shared_ptr<detail::AllocationState> state_;
    friend class Runtime;
};

struct View {
    std::shared_ptr<Allocation> allocation;
    std::size_t offset = 0;
    std::size_t bytes = 0;
};

// Immutable snapshot of every prerequisite in a logical stream's prefix.
// An empty/default Event is already complete. Re-recording an outer Torch
// Event replaces its Event value; previous snapshots remain unchanged.
class Event final {
public:
    Event() = default;
private:
    explicit Event(std::shared_ptr<detail::EventState>);
    std::shared_ptr<detail::EventState> state_;
    friend class Runtime;
};

class PreparedCsd final {
public:
    ~PreparedCsd();
private:
    explicit PreparedCsd(std::shared_ptr<Allocation>);
    std::shared_ptr<Allocation> code_;
    friend class Runtime;
};

// Physical CSD scheduling parameters, independent of a program's uniform ABI.
// The initial provider accepts this default single-workgroup geometry only.
// New geometry support extends validation/encoding without changing submission
// or the lifetime/dependency contract. Unsupported values reject pre-submit.
struct CsdLaunch {
    std::array<std::uint32_t, 3> workgroup_count{{1, 1, 1}};
    std::uint32_t workgroup_size = 16;
    std::uint32_t workgroups_per_supergroup = 16;
    std::uint32_t batch_count = 1;
};

class Runtime final {
public:
    static std::shared_ptr<Runtime> open(const Options&);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    std::shared_ptr<Allocation> allocate(std::size_t bytes);
    StreamId new_stream();                 // stream 0 is the default stream
    Event record(StreamId);
    void wait_event(StreamId, const Event&); // joins, never replaces, the prefix
    bool query(const Event&);
    bool query_stream(StreamId);
    void synchronize(const Event&);
    void synchronize_stream(StreamId);
    void synchronize_device();
    void collect();                         // nonblocking safe retirement
    void retain_until(StreamId, std::shared_ptr<Allocation>);

    // Four-byte-aligned contiguous byte views. Copy is bit-preserving for
    // FP32/int32/int64. Overlapping nonidentical ranges are rejected.
    void copy(StreamId, const View& dst, const View& src);
    void fill32(StreamId, const View& dst, std::uint32_t bits);

    // Provider preparation borrows canonical storage; it does not create a
    // second allocator. The caller owns the returned CLOEXEC DMA-BUF fd and
    // must retain the View/Allocation while preparing or holding an import.
    DrmDeviceIdentity device_identity() const;
    int export_dma_buf(const View&);

    // Joins stream and allocation frontiers, owns the complete invocation
    // before its first possible submission, and publishes its tail before
    // returning. Every externally accessed allocation must be in uses.
    void submit_provider(StreamId, const std::vector<View>& uses,
                         const std::shared_ptr<ProviderSubmission>&);

    // A provider can discover fatal device state during preparation/import,
    // before submit_provider owns its new invocation. Route that observation
    // through the same native containment authority so earlier accepted work
    // remains owned. This method always throws FatalError after containment.
    // Call only outside Runtime callbacks; a submit/retire/completion callback
    // instead throws ProviderFatalError, without allocating diagnostics, and
    // Runtime contains it under its existing lock.
    [[noreturn]] void report_provider_failure(const char* reason);

    std::shared_ptr<PreparedCsd> prepare_csd(const std::vector<std::uint64_t>& words);
    // Uniform words may contain checked allocation.gpu_address(offset) values.
    // Every GPU-referenced allocation must appear in borrowed; all are retained.
    void submit_csd(StreamId, const std::shared_ptr<PreparedCsd>&,
                    const std::vector<std::uint32_t>& uniforms,
                    const std::vector<View>& borrowed,
                    const CsdLaunch& launch = {});

    bool poisoned() const noexcept;
    std::string failure_reason() const;
    // Drain before normal shutdown. On failure the context is quarantined and
    // owners are retained. No destructor throws or treats timeout as cancel.
    void close();
private:
    explicit Runtime(std::shared_ptr<detail::State>);
    std::shared_ptr<detail::State> state_;
};

} // namespace mstar::pi
