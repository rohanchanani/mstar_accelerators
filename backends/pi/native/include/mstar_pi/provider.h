#pragma once

#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>

namespace mstar::pi {

// Allocation-free signal for a fatal backend observation inside a Runtime
// callback. In particular, device loss discovered before this invocation's
// first submit must not become an ordinary rejection if diagnostic allocation
// fails. Runtime remains responsible for containment and the outward error.
class ProviderFatalError final : public std::exception {
public:
    const char* what() const noexcept override {
        return "provider reported fatal device state";
    }
};

// A provider arms this marker immediately before its first call which might
// accept work. It must remain armed through all intermediate submissions and
// completion export. No exception after that point proves that work stopped.
class SubmissionAcceptance final {
public:
    SubmissionAcceptance() = default;
    SubmissionAcceptance(const SubmissionAcceptance&) = delete;
    SubmissionAcceptance& operator=(const SubmissionAcceptance&) = delete;
    void may_submit() noexcept { possible_ = true; }
    bool possible() const noexcept { return possible_; }
private:
    bool possible_ = false;
};

// The completion owns its actual backend device and receipt. It is independent
// of the mutable invocation lease: old Event snapshots may keep this object
// alive after the invocation's scratch and command resources become reusable.
// These methods must not call back into Runtime. query is nonblocking; wait
// returns on completion OR on expiration of the supplied duration. A backend
// error throws; neither a timeout nor an error authorizes resource retirement.
// query may run concurrently with wait from another host thread. Destruction
// must not submit new work or call Runtime; it can occur under Runtime locks.
class ProviderCompletion {
public:
    virtual ~ProviderCompletion() = default;
    virtual bool query() = 0;
    virtual void wait(std::chrono::nanoseconds duration) = 0;
};

struct ProviderReceipt {
    // Ownership transfers to Runtime on successful return from submit(). -1 is
    // only the Vulkan SYNC_FD already-signaled-payload sentinel, not a missing
    // receipt. completion is required even in that case. Any smaller value is
    // invalid. On an exception the provider retains its unreturned receipts.
    int sync_file = -1;
    std::shared_ptr<ProviderCompletion> completion;
};

class ProviderSubmission {
public:
    virtual ~ProviderSubmission() = default;

    // The object owns all mutable invocation resources BEFORE Runtime calls
    // this method. Runtime retains it until successful retirement or forever
    // under fatal quarantine. Submission runs under Runtime's ordering lock:
    // it must not reenter Runtime, allocate through Torch, or wait for GPU
    // completion. dependency_fd is borrowed for this call only; duplicate it
    // before a consuming Vulkan import. A negative fd means no device wait.
    // The returned tail must include every participating queue and ownership
    // release, without untracked deferred host operations or later untracked
    // submissions. Destruction also occurs under Runtime locks and must not
    // reenter Runtime. It may recycle the invocation only after Pending retires.
    virtual ProviderReceipt submit(int dependency_fd,
                                   SubmissionAcceptance& acceptance) = 0;

    // Called once, after the tail, every original prerequisite, and native
    // health have succeeded, before Runtime releases this invocation. Reusable
    // command state can be reset here with checked errors instead of inside a
    // destructor. Failure poisons Runtime and retains the invocation. This
    // method must not submit work, wait, or reenter Runtime.
    virtual void retire() {}
};

struct DrmDeviceIdentity {
    std::uint32_t major = 0;
    std::uint32_t minor = 0;
};

} // namespace mstar::pi
