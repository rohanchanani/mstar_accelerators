#include "mstar_pi/runtime.h"
#include "mstar_pi/detail/stream_prefixes.h"
#include "mstar_pi/detail/csd_launch.h"
#include "mstar_pi/detail/allocation_frontier.h"

#include <vulkan/vulkan.h>
#include <xf86drm.h>
#include <v3d_drm.h>
#include <linux/sync_file.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <utility>

namespace mstar::pi {
namespace detail {

class Fd {
public:
    Fd() = default;
    explicit Fd(int value) : value_(value) {}
    ~Fd() { if (value_ >= 0) ::close(value_); }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    Fd(Fd&& other) noexcept : value_(other.release()) {}
    Fd& operator=(Fd&& other) noexcept {
        if (this != &other) { Fd old(value_); value_ = other.release(); }
        return *this;
    }
    int get() const noexcept { return value_; }
    int release() noexcept { int v = value_; value_ = -1; return v; }
    Fd duplicate() const {
        if (value_ < 0) return {};
        int result = ::fcntl(value_, F_DUPFD_CLOEXEC, 0);
        if (result < 0) throw Rejected(std::string("duplicate native fd: ") + std::strerror(errno));
        return Fd(result);
    }
private:
    int value_ = -1;
};

static std::string system_error(const char* what) {
    return std::string(what) + ": " + std::strerror(errno);
}

static void check_vk_initial(VkResult result, const char* what) {
    if (result != VK_SUCCESS)
        throw Rejected(std::string(what) + " failed with VkResult " + std::to_string(result));
}

static std::uint64_t parameter(int fd, std::uint32_t key) {
    drm_v3d_get_param args{};
    args.param = key;
    if (::ioctl(fd, DRM_IOCTL_V3D_GET_PARAM, &args) != 0)
        throw Rejected(system_error("DRM_IOCTL_V3D_GET_PARAM"));
    return args.value;
}

// These values are from the matching Raspberry Pi kernel UAPI in r2. Older
// distro libdrm headers need not name these newer parameters yet.
constexpr std::uint32_t global_reset_parameter = 15;
constexpr std::uint32_t context_reset_parameter = 16;
// Private, explicitly versioned fork ABI supplied by kernel/ in this handoff.
// Stock kernels reject these IDs; there is deliberately no unsafe override.
constexpr std::uint32_t completion_contract_parameter = 0x50490001;
constexpr std::uint32_t device_status_parameter = 0x50490002;

static std::pair<std::uint32_t, std::uint32_t> csd_device_identity(int fd) {
    const auto ident0 = parameter(fd, DRM_V3D_PARAM_V3D_CORE0_IDENT0);
    const auto ident1 = parameter(fd, DRM_V3D_PARAM_V3D_CORE0_IDENT1);
    const auto hub_ident3 = parameter(fd, DRM_V3D_PARAM_V3D_HUB_IDENT3);
    return {static_cast<std::uint32_t>(((ident0 >> 24) & 0xff) * 10 + (ident1 & 0xf)),
            static_cast<std::uint32_t>((hub_ident3 >> 8) & 0xff)};
}

struct Context {
    Fd drm;
    DrmDeviceIdentity drm_identity;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    VkPhysicalDeviceMemoryProperties memory{};
    VkPhysicalDeviceProperties properties{};
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    PFN_vkGetMemoryFdPropertiesKHR memory_fd_properties = nullptr;
    PFN_vkImportSemaphoreFdKHR import_semaphore = nullptr;
    PFN_vkGetSemaphoreFdKHR export_semaphore = nullptr;
    std::uint64_t initial_global_reset = 0;
    std::uint64_t initial_context_reset = 0;
    std::uint32_t v3d_version = 0;
    std::uint32_t v3d_revision = 0;
    bool external_dedicated = false;
    // All native destruction is externally synchronized with this mutex.
    // State's mutex is acquired before Context's mutex when both are needed.
    std::recursive_mutex native_mutex;

    ~Context() {
        // Every accepted operation is represented by Pending and retired only
        // after success. A poisoned State deliberately retains this Context.
        if (device) {
            if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
            if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            if (descriptor_layout) vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
            if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

struct Point;

struct AllocationState {
    std::shared_ptr<Context> context;
    std::uint64_t id = 0;
    std::size_t logical_size = 0;
    std::size_t backing_size = 0;
    void* mapping = MAP_FAILED;
    std::uint32_t handle = 0;
    std::uint32_t gpu = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    AllocationFrontier<Point> ownership;
    ~AllocationState() {
        if (!context) return;
        std::lock_guard<std::recursive_mutex> lock(context->native_mutex);
        if (buffer) vkDestroyBuffer(context->device, buffer, nullptr);
        if (memory) vkFreeMemory(context->device, memory, nullptr);
        if (mapping != MAP_FAILED) ::munmap(mapping, backing_size);
        if (handle) {
            drm_gem_close close{};
            close.handle = handle;
            ::ioctl(context->drm.get(), DRM_IOCTL_GEM_CLOSE, &close);
        }
    }
};

struct Point {
    std::shared_ptr<Context> context;
    Fd sync_file;
    VkFence vk_fence = VK_NULL_HANDLE;
    std::shared_ptr<ProviderCompletion> provider_completion;
    bool receipt_valid = false;
    bool sync_file_ready = false;
    bool success = false; // only mutated under State::mutex
    ~Point() {
        if (vk_fence) {
            std::lock_guard<std::recursive_mutex> lock(context->native_mutex);
            vkDestroyFence(context->device, vk_fence, nullptr);
        }
    }
};

using Prefix = std::vector<std::shared_ptr<Point>>;

struct EventState {
    std::shared_ptr<Context> context;
    Prefix prefix;
};

struct Pending {
    std::shared_ptr<Context> context;
    std::shared_ptr<Point> point;
    Prefix prerequisites;
    std::vector<std::shared_ptr<AllocationState>> allocations;
    // Unlike Point's immutable completion owner, this is the mutable command,
    // descriptor, import and scratch lease. Retire it with accepted work even
    // when an old Event continues to retain its completion snapshot.
    std::shared_ptr<ProviderSubmission> provider_submission;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkSemaphore incoming = VK_NULL_HANDLE;
    VkSemaphore outgoing = VK_NULL_HANDLE;
    std::uint32_t in_sync = 0;
    std::uint32_t out_sync = 0;
    bool retirement_ready = false;
    ~Pending() {
        if (!context) return;
        std::lock_guard<std::recursive_mutex> lock(context->native_mutex);
        if (command) vkFreeCommandBuffers(context->device, context->command_pool, 1, &command);
        if (descriptors) vkDestroyDescriptorPool(context->device, descriptors, nullptr);
        if (incoming) vkDestroySemaphore(context->device, incoming, nullptr);
        if (outgoing) vkDestroySemaphore(context->device, outgoing, nullptr);
        if (in_sync) drmSyncobjDestroy(context->drm.get(), in_sync);
        if (out_sync) drmSyncobjDestroy(context->drm.get(), out_sync);
    }
};

struct HeldUse {
    Prefix prefix;
    std::shared_ptr<AllocationState> allocation;
    bool retirement_ready = false;
};

struct State : std::enable_shared_from_this<State> {
    std::shared_ptr<Context> context;
    Options options;
    mutable std::mutex mutex;
    StreamPrefixes<Point> streams;
    std::vector<std::shared_ptr<Pending>> pending;
    std::vector<HeldUse> held;
    std::uint64_t next_allocation = 1;
    bool closed = false;
    bool drained = false;
    bool fatal = false;
    std::string reason;
    std::shared_ptr<State> quarantine_self;

    void quarantine_noexcept(const char* why) noexcept {
        // Taking this existing shared ownership requires no new allocation.
        // No static destructor later destroys uncertain GPU owners. Kernel
        // quarantine also protects them if the process terminates.
        fatal = true;
        if (!quarantine_self) quarantine_self = shared_from_this();
        if (reason.empty()) {
            try { reason = why; } catch (...) { /* ownership is already safe */ }
        }
    }

    [[noreturn]] void poison(const char* why) {
        quarantine_noexcept(why);
        throw FatalError(reason.empty() ? "Pi context poisoned; native owners retained" : reason);
    }

    [[noreturn]] void poison_status(const char* why, int status) {
        // Containment must precede diagnostic formatting: allocation failure
        // must not turn a native failure into a recoverable host exception.
        quarantine_noexcept(why);
        try { reason += " (status " + std::to_string(status) + ")"; }
        catch (...) { /* the original reason and retained owners suffice */ }
        throw FatalError(reason.empty() ? "Pi native failure; owners retained" : reason);
    }

    void healthy(bool permit_closed = false) {
        if (fatal) throw FatalError(reason);
        if (closed && !permit_closed) throw Rejected("Pi runtime is closed");
    }

    void check_reset() {
        healthy(true);
        std::uint64_t global = 0, local = 0, status = 0;
        try {
            global = parameter(context->drm.get(), global_reset_parameter);
            local = parameter(context->drm.get(), context_reset_parameter);
            status = parameter(context->drm.get(), device_status_parameter);
        } catch (...) {
            // Even formatting an ioctl error can fail. Any inability to
            // establish health must contain owners before propagating failure.
            poison("cannot establish device epoch/health");
        }
        if (status != 0)
            poison("kernel reports sticky fatal CSD device status; reboot recovery required");
        if (global != context->initial_global_reset || local != context->initial_context_reset)
            poison("V3D reset epoch changed; state may have been replayed, owners retained");
    }

    void vk_check(VkResult value, const char* operation) {
        if (value == VK_SUCCESS) return;
        if (value == VK_ERROR_DEVICE_LOST) poison_status(operation, value);
        const std::string message = std::string(operation) + " failed with VkResult " + std::to_string(value);
        throw Rejected(message);
    }

    Prefix& stream(StreamId id) {
        return streams.at(id);
    }

    bool point_ready(const std::shared_ptr<Point>& p) {
        if (!p->receipt_valid) poison("accepted work has no complete native receipt");
        if (p->success) return true;
        if (p->sync_file.get() >= 0 && !p->sync_file_ready) {
            sync_file_info info{};
            if (::ioctl(p->sync_file.get(), SYNC_IOC_FILE_INFO, &info) != 0)
                poison_status("query sync_file status failed", errno);
            if (info.status < 0)
                poison_status("native completion reports execution error; settlement is not inferred from an error fence", info.status);
            if (info.status == 0) return false;
            p->sync_file_ready = true;
        }
        if (p->vk_fence) {
            const auto result = vkGetFenceStatus(context->device, p->vk_fence);
            if (result == VK_NOT_READY) return false;
            if (result != VK_SUCCESS)
                poison_status("Vulkan completion failed", result);
        }
        if (p->provider_completion) {
            bool ready = false;
            try { ready = p->provider_completion->query(); }
            catch (...) { poison("provider completion query failed; owners retained"); }
            if (!ready) return false;
        }
        p->success = true;
        std::atomic_thread_fence(std::memory_order_acquire);
        return true;
    }

    bool prefix_ready(const Prefix& prefix) {
        bool ready = true;
        // Query every prerequisite even if an earlier one is pending: errors
        // must not be hidden behind unrelated unfinished work.
        for (const auto& p : prefix) if (!point_ready(p)) ready = false;
        return ready;
    }

    void wait_prefix(const Prefix& prefix, std::unique_lock<std::mutex>& lock) {
        const auto deadline = std::chrono::steady_clock::now() + options.wait_timeout;
        for (const auto& p : prefix) {
            while (!point_ready(p)) {
                const auto now = std::chrono::steady_clock::now();
                if (now >= deadline) poison("native completion deadline expired; work is not cancelled");
                const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
                const auto ms = static_cast<int>(std::min<std::int64_t>(INT_MAX, std::max<std::int64_t>(1, remaining.count())));
                if (p->sync_file.get() >= 0 && !p->sync_file_ready) {
                    pollfd item{p->sync_file.get(), POLLIN, 0};
                    // Waiting must not prevent another host thread from
                    // enqueueing independent work on the same device.
                    lock.unlock();
                    const int result = ::poll(&item, 1, ms);
                    const int saved_errno = errno;
                    lock.lock();
                    if (result < 0 && saved_errno != EINTR) poison("poll native completion failed");
                    if (result > 0 && (item.revents & POLLNVAL)) poison("invalid native completion fd");
                } else if (p->vk_fence) {
                    const auto ns = static_cast<std::uint64_t>(ms) * 1000000ULL;
                    lock.unlock();
                    const auto result = vkWaitForFences(context->device, 1, &p->vk_fence, VK_TRUE, ns);
                    lock.lock();
                    if (result != VK_SUCCESS && result != VK_TIMEOUT)
                        poison_status("Vulkan wait failed", result);
                } else if (p->provider_completion) {
                    // The completion retains the actual provider device. A
                    // fence from that device cannot be waited through the
                    // native transfer Context's VkDevice.
                    const auto duration = std::chrono::milliseconds(ms);
                    lock.unlock();
                    bool failed = false;
                    try { p->provider_completion->wait(duration); }
                    catch (...) { failed = true; }
                    lock.lock();
                    if (failed) poison("provider completion wait failed; owners retained");
                } else {
                    poison("accepted submission has no completion receipt");
                }
                check_reset();
            }
        }
        check_reset();
    }

    void collect_locked() {
        check_reset();
        for (const auto& work : pending) {
            // Missing receipt occurs only on a poisoned submission path.
            if (!work->point) poison("pending native work has no completion receipt");
            const bool final_ready = point_ready(work->point);
            const bool inputs_ready = prefix_ready(work->prerequisites);
            work->retirement_ready = final_ready && inputs_ready;
        }
        for (auto& use : held) use.retirement_ready = prefix_ready(use.prefix);
        // Kernel v1 may report a late device error even after a fence became
        // positive. Establish health after inspecting every candidate and
        // before releasing any owner; failure above or here retains them all.
        check_reset();
        for (const auto& work : pending) {
            if (work->retirement_ready && work->provider_submission) {
                try { work->provider_submission->retire(); }
                catch (...) { poison("provider retirement failed; owners retained"); }
            }
        }
        pending.erase(std::remove_if(pending.begin(), pending.end(),
                                     [](const auto& work) { return work->retirement_ready; }),
                      pending.end());
        held.erase(std::remove_if(held.begin(), held.end(),
                                  [](const auto& use) { return use.retirement_ready; }),
                   held.end());
    }

    Fd merged(const Prefix& prefix) {
        Fd result;
        for (const auto& p : prefix) {
            if (point_ready(p)) continue;
            // Every pending submission exports a real sync_file or Vulkan's
            // already-signaled (-1) semaphore-payload sentinel.
            if (p->sync_file.get() < 0) {
                // SYNC_FD=-1 denotes an already-signaled semaphore payload.
                // Its VkFence may still lag in host bookkeeping. The payload
                // needs no wait, but Pending still retains the Vk submission
                // resources until the fence also completes.
                if (p->receipt_valid && (p->vk_fence || p->provider_completion)) continue;
                poison("pending provider work has no exportable dependency");
            }
            if (result.get() < 0) { result = p->sync_file.duplicate(); continue; }
            sync_merge_data merge{};
            std::strncpy(merge.name, "mstar-pi-join", sizeof(merge.name) - 1);
            merge.fd2 = p->sync_file.get();
            if (::ioctl(result.get(), SYNC_IOC_MERGE, &merge) != 0)
                throw Rejected(system_error("merge native dependencies"));
            Fd combined(merge.fence);
            if (::fcntl(combined.get(), F_SETFD, FD_CLOEXEC) != 0)
                throw Rejected(system_error("set merged fence CLOEXEC"));
            result = std::move(combined);
        }
        return result;
    }
};

static bool has_extension(VkPhysicalDevice physical, const char* name) {
    std::uint32_t count = 0;
    check_vk_initial(vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr), "enumerate device extensions");
    std::vector<VkExtensionProperties> list(count);
    check_vk_initial(vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, list.data()), "enumerate device extensions");
    return std::any_of(list.begin(), list.end(), [&](const auto& x) { return std::strcmp(x.extensionName, name) == 0; });
}

static std::shared_ptr<Context> create_context(const Options& options) {
    auto context = std::make_shared<Context>();
    context->drm = Fd(::open(options.render_node.c_str(), O_RDWR | O_CLOEXEC));
    if (context->drm.get() < 0) throw Rejected(system_error("open V3D render node"));
    struct stat file{};
    if (::fstat(context->drm.get(), &file) != 0 || !S_ISCHR(file.st_mode))
        throw Rejected("render_node must identify a DRM character device");
    context->drm_identity = {static_cast<std::uint32_t>(major(file.st_rdev)),
                             static_cast<std::uint32_t>(minor(file.st_rdev))};
    std::uint64_t contract = 0;
    try { contract = parameter(context->drm.get(), completion_contract_parameter); }
    catch (const Rejected&) {
        throw Rejected("V3D completion contract v1 is unavailable; install the reviewed matching kernel patch before native execution");
    }
    if (contract != 1) throw Rejected("unsupported V3D completion contract version; exactly version 1 is required");
    if (parameter(context->drm.get(), device_status_parameter) != 0)
        throw FatalError("V3D device is already quarantined; reboot recovery is required");
    if (parameter(context->drm.get(), DRM_V3D_PARAM_SUPPORTS_CSD) == 0)
        throw Rejected("selected DRM device does not support CSD");
    const auto csd_identity = csd_device_identity(context->drm.get());
    context->v3d_version = csd_identity.first;
    context->v3d_revision = csd_identity.second;
    if (!supports_csd_device(context->v3d_version, context->v3d_revision))
        throw Rejected("native provider currently requires V3D 7.1 revision 6 or later");
    context->initial_global_reset = parameter(context->drm.get(), global_reset_parameter);
    context->initial_context_reset = parameter(context->drm.get(), context_reset_parameter);

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "mstar-pi";
    app.apiVersion = VK_API_VERSION_1_2;
    const char* layer = "VK_LAYER_KHRONOS_validation";
    if (options.validation) {
        std::uint32_t count = 0;
        check_vk_initial(vkEnumerateInstanceLayerProperties(&count, nullptr), "enumerate layers");
        std::vector<VkLayerProperties> layers(count);
        check_vk_initial(vkEnumerateInstanceLayerProperties(&count, layers.data()), "enumerate layers");
        if (std::none_of(layers.begin(), layers.end(), [&](const auto& v) { return std::strcmp(v.layerName, layer) == 0; }))
            throw Rejected("requested Vulkan validation layer is unavailable");
    }
    VkInstanceCreateInfo instance{};
    instance.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance.pApplicationInfo = &app;
    instance.enabledLayerCount = options.validation ? 1 : 0;
    instance.ppEnabledLayerNames = options.validation ? &layer : nullptr;
    check_vk_initial(vkCreateInstance(&instance, nullptr, &context->instance), "create Vulkan instance");
    std::uint32_t count = 0;
    check_vk_initial(vkEnumeratePhysicalDevices(context->instance, &count, nullptr), "enumerate physical devices");
    std::vector<VkPhysicalDevice> devices(count);
    check_vk_initial(vkEnumeratePhysicalDevices(context->instance, &count, devices.data()), "enumerate physical devices");
    for (auto candidate : devices) {
        if (!has_extension(candidate, VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME)) continue;
        VkPhysicalDeviceDrmPropertiesEXT drm{};
        drm.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT;
        VkPhysicalDeviceProperties2 properties{};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties.pNext = &drm;
        vkGetPhysicalDeviceProperties2(candidate, &properties);
        if (drm.hasRender && drm.renderMajor == ::major(file.st_rdev) && drm.renderMinor == ::minor(file.st_rdev)) {
            context->physical = candidate;
            context->properties = properties.properties;
            break;
        }
    }
    if (!context->physical) throw Rejected("no Vulkan physical device matches the selected DRM render node");
    const char* extensions[] = {
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME,
    };
    for (const char* extension : extensions)
        if (!has_extension(context->physical, extension)) throw Rejected(std::string("missing Vulkan extension: ") + extension);

    VkPhysicalDeviceExternalBufferInfo buffer{};
    buffer.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkExternalBufferProperties external{};
    external.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
    vkGetPhysicalDeviceExternalBufferProperties(context->physical, &buffer, &external);
    if (!(external.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT))
        throw Rejected("Vulkan storage buffers cannot import DMA-BUF on this target");
    context->external_dedicated = external.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT;
    VkPhysicalDeviceExternalSemaphoreInfo semaphore{};
    semaphore.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO;
    semaphore.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    VkExternalSemaphoreProperties semaphore_properties{};
    semaphore_properties.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES;
    vkGetPhysicalDeviceExternalSemaphoreProperties(context->physical, &semaphore, &semaphore_properties);
    constexpr auto needed = VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT | VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT;
    if ((semaphore_properties.externalSemaphoreFeatures & needed) != needed)
        throw Rejected("Vulkan requires both SYNC_FD import and export");

    std::uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(context->physical, &families, nullptr);
    std::vector<VkQueueFamilyProperties> queue_properties(families);
    vkGetPhysicalDeviceQueueFamilyProperties(context->physical, &families, queue_properties.data());
    bool found = false;
    for (std::uint32_t i = 0; i < families; ++i) if (queue_properties[i].queueCount && (queue_properties[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
        context->family = i; found = true; break;
    }
    if (!found) throw Rejected("Vulkan device has no compute queue");
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue{};
    queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue.queueFamilyIndex = context->family;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    VkDeviceCreateInfo device{};
    device.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device.queueCreateInfoCount = 1;
    device.pQueueCreateInfos = &queue;
    device.enabledExtensionCount = sizeof(extensions) / sizeof(*extensions);
    device.ppEnabledExtensionNames = extensions;
    check_vk_initial(vkCreateDevice(context->physical, &device, nullptr, &context->device), "create Vulkan device");
    vkGetDeviceQueue(context->device, context->family, 0, &context->queue);
    vkGetPhysicalDeviceMemoryProperties(context->physical, &context->memory);
    context->memory_fd_properties = reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(vkGetDeviceProcAddr(context->device, "vkGetMemoryFdPropertiesKHR"));
    context->import_semaphore = reinterpret_cast<PFN_vkImportSemaphoreFdKHR>(vkGetDeviceProcAddr(context->device, "vkImportSemaphoreFdKHR"));
    context->export_semaphore = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(vkGetDeviceProcAddr(context->device, "vkGetSemaphoreFdKHR"));
    if (!context->memory_fd_properties || !context->import_semaphore || !context->export_semaphore)
        throw Rejected("missing required Vulkan external-memory/synchronization entry points");

    VkCommandPoolCreateInfo pool{};
    pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.queueFamilyIndex = context->family;
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check_vk_initial(vkCreateCommandPool(context->device, &pool, nullptr, &context->command_pool), "create command pool");
    VkDescriptorSetLayoutBinding bindings[2]{};
    for (std::uint32_t i = 0; i < 2; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo descriptor{};
    descriptor.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor.bindingCount = 2;
    descriptor.pBindings = bindings;
    check_vk_initial(vkCreateDescriptorSetLayout(context->device, &descriptor, nullptr, &context->descriptor_layout), "create descriptor layout");
    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size = 5 * sizeof(std::uint32_t);
    VkPipelineLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &context->descriptor_layout;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &push;
    check_vk_initial(vkCreatePipelineLayout(context->device, &layout, nullptr, &context->pipeline_layout), "create pipeline layout");
    std::ifstream shader(options.copy_shader, std::ios::binary | std::ios::ate);
    if (!shader) throw Rejected("cannot open transfer SPIR-V: " + options.copy_shader);
    const auto size = shader.tellg();
    if (size < 20 || size % 4 != 0) throw Rejected("transfer SPIR-V has invalid byte extent");
    std::vector<std::uint32_t> words(static_cast<std::size_t>(size) / 4);
    shader.seekg(0);
    shader.read(reinterpret_cast<char*>(words.data()), size);
    if (!shader || words[0] != 0x07230203u) throw Rejected("transfer SPIR-V is not a complete module");
    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = words.size() * 4;
    module_info.pCode = words.data();
    VkShaderModule module = VK_NULL_HANDLE;
    check_vk_initial(vkCreateShaderModule(context->device, &module_info, nullptr, &module), "create transfer shader module");
    VkComputePipelineCreateInfo pipeline{};
    pipeline.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline.layout = context->pipeline_layout;
    pipeline.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline.stage.module = module;
    pipeline.stage.pName = "main";
    const auto result = vkCreateComputePipelines(context->device, VK_NULL_HANDLE, 1, &pipeline, nullptr, &context->pipeline);
    vkDestroyShaderModule(context->device, module, nullptr);
    check_vk_initial(result, "create transfer pipeline");
    return context;
}

static std::shared_ptr<AllocationState> allocate_locked(State& state, std::size_t bytes) {
    auto context = state.context;
    if (bytes > std::numeric_limits<std::uint32_t>::max() - 4096ULL)
        throw Rejected("allocation exceeds this provider's 32-bit GPU address/extent domain");
    const std::size_t buffer_size = std::max<std::size_t>(4, (bytes + 3) & ~std::size_t(3));
    if (buffer_size > context->properties.limits.maxStorageBufferRange)
        throw Rejected("allocation exceeds Vulkan maxStorageBufferRange for this provider");
    auto allocation = std::make_shared<AllocationState>();
    allocation->context = context;
    allocation->id = state.next_allocation++;
    allocation->logical_size = bytes;
    VkExternalMemoryBufferCreateInfo external{};
    external.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkBufferCreateInfo buffer{};
    buffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer.pNext = &external;
    buffer.size = buffer_size;
    buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    state.vk_check(vkCreateBuffer(context->device, &buffer, nullptr, &allocation->buffer), "create imported buffer view");
    VkMemoryDedicatedRequirements dedicated{};
    dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
    VkMemoryRequirements2 requirements{};
    requirements.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    requirements.pNext = &dedicated;
    VkBufferMemoryRequirementsInfo2 info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
    info.buffer = allocation->buffer;
    vkGetBufferMemoryRequirements2(context->device, &info, &requirements);
    const auto required = requirements.memoryRequirements.size;
    if (required > std::numeric_limits<std::uint32_t>::max())
        throw Rejected("native backing requirement exceeds DRM CREATE_BO size");
    drm_v3d_create_bo create{};
    create.size = static_cast<std::uint32_t>(required);
    if (::ioctl(context->drm.get(), DRM_IOCTL_V3D_CREATE_BO, &create) != 0)
        throw Rejected(system_error("allocate DRM BO"));
    allocation->handle = create.handle;
    allocation->gpu = create.offset;
    // CREATE_BO may round the requested Vulkan extent to the kernel page
    // size. The exported DMA-BUF is authoritative for physical backing size;
    // logical bytes and VkBuffer requirements remain separate facts.
    int prime = -1;
    if (drmPrimeHandleToFD(context->drm.get(), allocation->handle, DRM_CLOEXEC | DRM_RDWR, &prime) != 0)
        throw Rejected(system_error("export DRM BO as DMA-BUF"));
    Fd dma_buf(prime);
    const off_t backing = ::lseek(dma_buf.get(), 0, SEEK_END);
    if (backing < 0) throw Rejected(system_error("query DRM BO backing size"));
    if (static_cast<std::uint64_t>(backing) < required ||
        static_cast<std::uint64_t>(backing) > std::numeric_limits<std::size_t>::max())
        throw Rejected("DRM BO backing extent cannot satisfy Vulkan requirements");
    allocation->backing_size = static_cast<std::size_t>(backing);
    if (std::uint64_t(allocation->gpu) + buffer_size > std::uint64_t(std::numeric_limits<std::uint32_t>::max()) + 1)
        throw Rejected("DRM GPU address range overflows 32-bit addressing");
    drm_v3d_mmap_bo mapping{};
    mapping.handle = allocation->handle;
    if (::ioctl(context->drm.get(), DRM_IOCTL_V3D_MMAP_BO, &mapping) != 0)
        throw Rejected(system_error("map DRM BO offset"));
    allocation->mapping = ::mmap(nullptr, allocation->backing_size, PROT_READ | PROT_WRITE,
                                 MAP_SHARED, context->drm.get(), static_cast<off_t>(mapping.offset));
    if (allocation->mapping == MAP_FAILED) throw Rejected(system_error("mmap DRM BO"));

    VkMemoryFdPropertiesKHR properties{};
    properties.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR;
    state.vk_check(context->memory_fd_properties(context->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
                                                dma_buf.get(), &properties), "query DMA-BUF memory types");
    const auto types = properties.memoryTypeBits & requirements.memoryRequirements.memoryTypeBits;
    std::uint32_t memory_type = UINT32_MAX;
    constexpr auto host_memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (std::uint32_t i = 0; i < context->memory.memoryTypeCount; ++i)
        if ((types & (1u << i)) && (context->memory.memoryTypes[i].propertyFlags & host_memory) == host_memory) {
            memory_type = i; break;
        }
    if (memory_type == UINT32_MAX) throw Rejected("no coherent host-visible Vulkan memory type for this DRM allocation provider");
    VkMemoryDedicatedAllocateInfo dedicated_info{};
    dedicated_info.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated_info.buffer = allocation->buffer;
    VkImportMemoryFdInfoKHR import{};
    import.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
    import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    import.fd = dma_buf.get();
    if (context->external_dedicated || dedicated.requiresDedicatedAllocation) import.pNext = &dedicated_info;
    VkMemoryAllocateInfo memory{};
    memory.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memory.pNext = &import;
    memory.allocationSize = allocation->backing_size;
    memory.memoryTypeIndex = memory_type;
    state.vk_check(vkAllocateMemory(context->device, &memory, nullptr, &allocation->memory), "import DRM allocation into Vulkan");
    dma_buf.release(); // successful Vulkan import owns this fd
    state.vk_check(vkBindBufferMemory(context->device, allocation->buffer, allocation->memory, 0), "bind imported Vulkan buffer");
    return allocation;
}

static void validate_view(const State& state, const View& view, const std::shared_ptr<AllocationState>& allocation) {
    if (!allocation || allocation->context.get() != state.context.get())
        throw Rejected("native view belongs to another device context");
    if (view.offset > allocation->logical_size || view.bytes > allocation->logical_size - view.offset)
        throw Rejected("native view exceeds its logical allocation");
    if ((view.offset | view.bytes) & 3u) throw Rejected("this native provider requires four-byte-aligned views");
}

static void transfer_locked(State& state, StreamId stream_id,
                            const View& destination, const std::shared_ptr<AllocationState>& dst,
                            const View& source, const std::shared_ptr<AllocationState>& src,
                            bool fill, std::uint32_t value) {
    validate_view(state, destination, dst);
    validate_view(state, source, src);
    if (!fill && source.bytes != destination.bytes) throw Rejected("native copy extents differ");
    Prefix prerequisites = state.stream(stream_id);
    if (destination.bytes == 0) return;
    if (!fill && src.get() == dst.get()) {
        if (source.offset == destination.offset) return;
        if (source.offset < destination.offset + destination.bytes && destination.offset < source.offset + source.bytes)
            throw Rejected("overlapping native copy requires an explicit temporary");
    }
    auto context = state.context;
    const auto words = destination.bytes / 4;
    const auto groups = (words + 15) / 16;
    const auto groups_x = std::min<std::size_t>(groups, context->properties.limits.maxComputeWorkGroupCount[0]);
    const auto groups_y = (groups + groups_x - 1) / groups_x;
    if (groups_y > context->properties.limits.maxComputeWorkGroupCount[1]) throw Rejected("transfer dispatch exceeds target grid bounds");
    auto work = std::make_shared<Pending>();
    work->context = context;
    work->prerequisites = prerequisites;
    work->allocations = {src};
    if (dst.get() != src.get()) work->allocations.push_back(dst);
    for (const auto& allocation : work->allocations)
        allocation->ownership.append_dependency(work->prerequisites);
    Fd dependency = state.merged(work->prerequisites);

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo pool{};
    pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool.maxSets = 1;
    pool.poolSizeCount = 1;
    pool.pPoolSizes = &pool_size;
    state.vk_check(vkCreateDescriptorPool(context->device, &pool, nullptr, &work->descriptors), "create submission descriptors");
    VkDescriptorSetAllocateInfo set_info{};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = work->descriptors;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &context->descriptor_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    state.vk_check(vkAllocateDescriptorSets(context->device, &set_info, &set), "allocate submission descriptor set");
    VkDescriptorBufferInfo buffers[2] = {
        {src->buffer, 0, std::max<std::size_t>(4, (src->logical_size + 3) & ~std::size_t(3))},
        {dst->buffer, 0, std::max<std::size_t>(4, (dst->logical_size + 3) & ~std::size_t(3))},
    };
    VkWriteDescriptorSet writes[2]{};
    for (std::uint32_t i = 0; i < 2; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffers[i];
    }
    vkUpdateDescriptorSets(context->device, 2, writes, 0, nullptr);
    VkCommandBufferAllocateInfo command{};
    command.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command.commandPool = context->command_pool;
    command.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command.commandBufferCount = 1;
    state.vk_check(vkAllocateCommandBuffers(context->device, &command, &work->command), "allocate transfer command buffer");
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    state.vk_check(vkBeginCommandBuffer(work->command, &begin), "begin transfer commands");
    std::vector<VkBufferMemoryBarrier> barriers;
    for (const auto& allocation : work->allocations) {
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
        barrier.dstQueueFamilyIndex = context->family;
        barrier.buffer = allocation->buffer;
        barrier.size = VK_WHOLE_SIZE;
        barriers.push_back(barrier);
    }
    vkCmdPipelineBarrier(work->command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         static_cast<std::uint32_t>(barriers.size()), barriers.data(), 0, nullptr);
    VkMemoryBarrier host_writes{};
    host_writes.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    host_writes.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    host_writes.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(work->command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &host_writes, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(work->command, VK_PIPELINE_BIND_POINT_COMPUTE, context->pipeline);
    vkCmdBindDescriptorSets(work->command, VK_PIPELINE_BIND_POINT_COMPUTE, context->pipeline_layout, 0, 1, &set, 0, nullptr);
    std::uint32_t push[5] = {static_cast<std::uint32_t>(source.offset / 4),
                             static_cast<std::uint32_t>(destination.offset / 4),
                             static_cast<std::uint32_t>(words), value, fill ? 1u : 0u};
    vkCmdPushConstants(work->command, context->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    vkCmdDispatch(work->command, static_cast<std::uint32_t>(groups_x), static_cast<std::uint32_t>(groups_y), 1);
    VkMemoryBarrier host_reads{};
    host_reads.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    host_reads.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    host_reads.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(work->command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &host_reads, 0, nullptr, 0, nullptr);
    for (auto& barrier : barriers) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = 0;
        barrier.srcQueueFamilyIndex = context->family;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
    }
    vkCmdPipelineBarrier(work->command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr,
                         static_cast<std::uint32_t>(barriers.size()), barriers.data(), 0, nullptr);
    state.vk_check(vkEndCommandBuffer(work->command), "end transfer commands");

    VkSemaphoreCreateInfo semaphore{};
    semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    if (dependency.get() >= 0) {
        state.vk_check(vkCreateSemaphore(context->device, &semaphore, nullptr, &work->incoming), "create incoming semaphore");
        VkImportSemaphoreFdInfoKHR import{};
        import.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
        import.semaphore = work->incoming;
        import.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
        import.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
        import.fd = dependency.get();
        state.vk_check(context->import_semaphore(context->device, &import), "import transfer dependencies");
        dependency.release();
    }
    VkExportSemaphoreCreateInfo export_info{};
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    semaphore.pNext = &export_info;
    state.vk_check(vkCreateSemaphore(context->device, &semaphore, nullptr, &work->outgoing), "create outgoing semaphore");
    auto point = std::make_shared<Point>();
    point->context = context;
    VkFenceCreateInfo fence{};
    fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    state.vk_check(vkCreateFence(context->device, &fence, nullptr, &point->vk_fence), "create transfer fence");
    work->point = point;
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = work->incoming ? 1 : 0;
    submit.pWaitSemaphores = work->incoming ? &work->incoming : nullptr;
    submit.pWaitDstStageMask = work->incoming ? &wait_stage : nullptr;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &work->command;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &work->outgoing;
    // Allocate all C++ containers that can fail before handing work to hardware.
    Prefix successor{point};
    state.pending.push_back(work);
    std::atomic_thread_fence(std::memory_order_release);
    const auto accepted = vkQueueSubmit(context->queue, 1, &submit, point->vk_fence);
    if (accepted != VK_SUCCESS)
        state.poison("Vulkan submission failed after acceptance became possible");
    VkSemaphoreGetFdInfoKHR get{};
    get.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    get.semaphore = work->outgoing;
    get.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    int exported = -2;
    const auto exported_result = context->export_semaphore(context->device, &get, &exported);
    if (exported_result != VK_SUCCESS || exported < -1)
        state.poison("accepted Vulkan work could not export its completion receipt");
    point->sync_file = Fd(exported); // -1 means an already-signaled payload
    point->receipt_valid = true;
    for (const auto& allocation : work->allocations) allocation->ownership.publish(point);
    state.stream(stream_id).swap(successor);
}

} // namespace detail

bool probe_device(const std::string& render_node) noexcept {
    try {
        detail::Fd fd(::open(render_node.c_str(), O_RDWR | O_CLOEXEC));
        if (fd.get() < 0) return false;
        struct stat file{};
        if (::fstat(fd.get(), &file) != 0 || !S_ISCHR(file.st_mode)) return false;
        if (detail::parameter(fd.get(), detail::completion_contract_parameter) != 1 ||
            detail::parameter(fd.get(), detail::device_status_parameter) != 0 ||
            detail::parameter(fd.get(), DRM_V3D_PARAM_SUPPORTS_CSD) == 0) return false;
        const auto identity = detail::csd_device_identity(fd.get());
        return detail::supports_csd_device(identity.first, identity.second);
    } catch (...) {
        return false;
    }
}

Allocation::Allocation(std::shared_ptr<detail::AllocationState> state) : state_(std::move(state)) {}
Allocation::~Allocation() = default;
std::size_t Allocation::size() const noexcept { return state_->logical_size; }
std::size_t Allocation::backing_size() const noexcept { return state_->backing_size; }
void* Allocation::mapped() const noexcept { return state_->mapping; }
std::uint64_t Allocation::identity() const noexcept { return state_->id; }
std::uint32_t Allocation::gpu_address(std::size_t offset) const {
    if (offset > state_->logical_size || std::uint64_t(state_->gpu) + offset > UINT32_MAX)
        throw Rejected("GPU address offset is outside the allocation");
    return state_->gpu + static_cast<std::uint32_t>(offset);
}
Event::Event(std::shared_ptr<detail::EventState> state) : state_(std::move(state)) {}
PreparedCsd::PreparedCsd(std::shared_ptr<Allocation> code) : code_(std::move(code)) {}
PreparedCsd::~PreparedCsd() = default;
Runtime::Runtime(std::shared_ptr<detail::State> state) : state_(std::move(state)) {}

std::shared_ptr<Runtime> Runtime::open(const Options& options) {
    if (options.wait_timeout.count() <= 0) throw Rejected("wait_timeout must be positive");
    if (options.copy_shader.empty()) throw Rejected("copy_shader must name the compiled transfer SPIR-V");
    auto state = std::make_shared<detail::State>();
    state->options = options;
    state->context = detail::create_context(options);
    return std::shared_ptr<Runtime>(new Runtime(std::move(state)));
}

Runtime::~Runtime() {
    try { close(); }
    catch (const std::exception& error) {
        state_->quarantine_noexcept("exception while draining runtime teardown");
        std::fprintf(stderr, "mstar_pi: runtime teardown retained an unsafe context: %s\n", error.what());
    }
}

std::shared_ptr<Allocation> Runtime::allocate(std::size_t bytes) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    std::lock_guard<std::recursive_mutex> native(state_->context->native_mutex);
    state_->healthy();
    state_->collect_locked();
    return std::shared_ptr<Allocation>(new Allocation(detail::allocate_locked(*state_, bytes)));
}

StreamId Runtime::new_stream() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->healthy();
    return state_->streams.create();
}

Event Runtime::record(StreamId id) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->healthy();
    auto event = std::make_shared<detail::EventState>();
    event->context = state_->context;
    event->prefix = state_->stream(id);
    return Event(std::move(event));
}

void Runtime::wait_event(StreamId id, const Event& event) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->healthy();
    state_->stream(id); // validate even an empty/default event
    if (!event.state_) return;
    if (event.state_->context.get() != state_->context.get()) throw Rejected("event belongs to another native device context");
    state_->streams.join(id, event.state_->prefix);
}

bool Runtime::query(const Event& event) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    std::lock_guard<std::recursive_mutex> native(state_->context->native_mutex);
    state_->healthy(true);
    state_->check_reset();
    if (!event.state_) return true;
    if (event.state_->context.get() != state_->context.get()) throw Rejected("event belongs to another native device context");
    const bool complete = state_->prefix_ready(event.state_->prefix);
    state_->collect_locked();
    return complete;
}

bool Runtime::query_stream(StreamId id) { return query(record(id)); }

void Runtime::synchronize(const Event& event) {
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->healthy(true);
    state_->check_reset();
    if (event.state_) {
        if (event.state_->context.get() != state_->context.get()) throw Rejected("event belongs to another native device context");
        state_->wait_prefix(event.state_->prefix, lock);
    }
    state_->collect_locked();
}

void Runtime::synchronize_stream(StreamId id) { synchronize(record(id)); }

void Runtime::synchronize_device() {
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->healthy(true);
    state_->check_reset();
    detail::Prefix prefix = state_->streams.snapshot_all();
    // A successful native call always updates a stream. Explicit held uses can
    // retain earlier prefixes, so settle them before device completion too.
    for (const auto& use : state_->held)
        prefix.insert(prefix.end(), use.prefix.begin(), use.prefix.end());
    state_->wait_prefix(prefix, lock);
    state_->collect_locked();
}

void Runtime::collect() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    std::lock_guard<std::recursive_mutex> native(state_->context->native_mutex);
    state_->healthy(true);
    state_->collect_locked();
}

void Runtime::retain_until(StreamId id, std::shared_ptr<Allocation> allocation) {
    if (!allocation) throw Rejected("cannot retain a null allocation");
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->healthy();
    if (allocation->state_->context.get() != state_->context.get()) throw Rejected("retained allocation belongs to another context");
    auto prefix = state_->stream(id);
    if (!prefix.empty()) state_->held.push_back({std::move(prefix), allocation->state_});
}

void Runtime::copy(StreamId id, const View& dst, const View& src) {
    if (!dst.allocation || !src.allocation) throw Rejected("copy requires allocation owners");
    std::lock_guard<std::mutex> lock(state_->mutex);
    std::lock_guard<std::recursive_mutex> native(state_->context->native_mutex);
    state_->healthy();
    state_->collect_locked();
    detail::transfer_locked(*state_, id, dst, dst.allocation->state_, src, src.allocation->state_, false, 0);
}

void Runtime::fill32(StreamId id, const View& dst, std::uint32_t value) {
    if (!dst.allocation) throw Rejected("fill requires an allocation owner");
    std::lock_guard<std::mutex> lock(state_->mutex);
    std::lock_guard<std::recursive_mutex> native(state_->context->native_mutex);
    state_->healthy();
    state_->collect_locked();
    detail::transfer_locked(*state_, id, dst, dst.allocation->state_, dst, dst.allocation->state_, true, value);
}

DrmDeviceIdentity Runtime::device_identity() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->healthy();
    return state_->context->drm_identity;
}

int Runtime::export_dma_buf(const View& view) {
    if (!view.allocation) throw Rejected("provider import requires an allocation owner");
    std::lock_guard<std::mutex> lock(state_->mutex);
    std::lock_guard<std::recursive_mutex> native(state_->context->native_mutex);
    state_->healthy();
    state_->check_reset();
    detail::validate_view(*state_, view, view.allocation->state_);
    int exported = -1;
    if (drmPrimeHandleToFD(state_->context->drm.get(), view.allocation->state_->handle,
                         DRM_CLOEXEC | DRM_RDWR, &exported) != 0)
        throw Rejected(detail::system_error("export provider DMA-BUF"));
    return exported;
}

void Runtime::submit_provider(StreamId id, const std::vector<View>& uses,
                              const std::shared_ptr<ProviderSubmission>& submission) {
    if (!submission) throw Rejected("provider submission requires an invocation owner");
    std::lock_guard<std::mutex> lock(state_->mutex);
    std::lock_guard<std::recursive_mutex> native(state_->context->native_mutex);
    state_->healthy();
    state_->collect_locked();

    auto work = std::make_shared<detail::Pending>();
    work->context = state_->context;
    work->provider_submission = submission;
    work->prerequisites = state_->stream(id);
    for (const auto& view : uses) {
        if (!view.allocation) throw Rejected("provider use has no allocation owner");
        detail::validate_view(*state_, view, view.allocation->state_);
        if (std::find(work->allocations.begin(), work->allocations.end(), view.allocation->state_) == work->allocations.end())
            work->allocations.push_back(view.allocation->state_);
    }
    for (const auto& allocation : work->allocations)
        allocation->ownership.append_dependency(work->prerequisites);
    detail::Fd dependency = state_->merged(work->prerequisites);
    auto point = std::make_shared<detail::Point>();
    point->context = state_->context;
    work->point = point;
    detail::Prefix successor{point};

    // Every fallible allocation needed to retain and publish this invocation
    // precedes the callback. The same ordering lock protects frontier snapshot,
    // all possibly accepted submissions, and tail publication. The callback is
    // explicitly forbidden from reentering Runtime or waiting for completion.
    state_->pending.push_back(work);
    SubmissionAcceptance acceptance;
    ProviderReceipt receipt;
    try {
        std::atomic_thread_fence(std::memory_order_release);
        receipt = submission->submit(dependency.get(), acceptance);
    } catch (const ProviderFatalError&) {
        state_->poison("provider reported fatal device state; owners retained");
    } catch (const FatalError&) {
        // A provider can discover device loss during preparation, before its
        // own first submit. It is still fatal for all users of this device.
        state_->poison("provider reported fatal device state; owners retained");
    } catch (...) {
        if (acceptance.possible())
            state_->poison("provider submission failed after possible acceptance; owners retained");
        // No callback can reenter the State, so this is still the last Pending.
        // Previous accepted work remains tracked; rejection does not drain it.
        state_->pending.pop_back();
        throw;
    }

    // Take ownership before inspecting the receipt so an invalid response
    // cannot leak an fd. Missing/invalid status after a possible submission is
    // contained, never reclassified as an unsupported graph.
    point->sync_file = detail::Fd(receipt.sync_file);
    point->provider_completion = std::move(receipt.completion);
    if (!acceptance.possible() || !point->provider_completion || receipt.sync_file < -1)
        state_->poison("provider returned an invalid submission receipt; owners retained");
    if (point->sync_file.get() >= 0 && ::fcntl(point->sync_file.get(), F_SETFD, FD_CLOEXEC) != 0)
        state_->poison("accepted provider completion could not set CLOEXEC; owners retained");
    point->receipt_valid = true;
    for (const auto& allocation : work->allocations) allocation->ownership.publish(point);
    state_->stream(id).swap(successor);
}

[[noreturn]] void Runtime::report_provider_failure(const char* reason) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->poison(reason ? reason : "provider reported fatal device state; owners retained");
}

std::shared_ptr<PreparedCsd> Runtime::prepare_csd(const std::vector<std::uint64_t>& words) {
    if (words.empty() || words.size() > UINT32_MAX / sizeof(std::uint64_t)) throw Rejected("invalid CSD program extent");
    auto code = allocate(words.size() * sizeof(std::uint64_t));
    std::memcpy(code->mapped(), words.data(), words.size() * sizeof(std::uint64_t));
    std::atomic_thread_fence(std::memory_order_release);
    return std::shared_ptr<PreparedCsd>(new PreparedCsd(std::move(code)));
}

void Runtime::submit_csd(StreamId id, const std::shared_ptr<PreparedCsd>& program,
                         const std::vector<std::uint32_t>& uniforms, const std::vector<View>& borrowed,
                         const CsdLaunch& launch) {
    if (!program) throw Rejected("CSD submission requires a prepared program");
    if (uniforms.size() > UINT32_MAX / 4) throw Rejected("CSD uniform block is too large");
    std::lock_guard<std::mutex> lock(state_->mutex);
    std::lock_guard<std::recursive_mutex> native(state_->context->native_mutex);
    state_->healthy();
    state_->collect_locked();
    const auto context = state_->context;
    const auto cfg = detail::encode_csd_launch(launch, context->v3d_version, context->v3d_revision);
    if (program->code_->state_->context.get() != context.get()) throw Rejected("CSD program belongs to another native context");
    auto work = std::make_shared<detail::Pending>();
    work->context = context;
    work->prerequisites = state_->stream(id);
    work->allocations.push_back(program->code_->state_);
    for (const auto& view : borrowed) {
        if (!view.allocation) throw Rejected("CSD borrow has no allocation owner");
        detail::validate_view(*state_, view, view.allocation->state_);
        if (std::find(work->allocations.begin(), work->allocations.end(), view.allocation->state_) == work->allocations.end())
            work->allocations.push_back(view.allocation->state_);
    }
    std::shared_ptr<detail::AllocationState> uniform_storage;
    if (!uniforms.empty()) {
        uniform_storage = detail::allocate_locked(*state_, uniforms.size() * 4);
        std::memcpy(uniform_storage->mapping, uniforms.data(), uniforms.size() * 4);
        work->allocations.push_back(uniform_storage);
    }
    for (const auto& allocation : work->allocations)
        allocation->ownership.append_dependency(work->prerequisites);
    std::vector<std::uint32_t> handles;
    handles.reserve(work->allocations.size());
    for (const auto& allocation : work->allocations) handles.push_back(allocation->handle);
    detail::Fd dependency = state_->merged(work->prerequisites);
    if (dependency.get() >= 0) {
        if (drmSyncobjCreate(context->drm.get(), 0, &work->in_sync) != 0)
            throw Rejected(detail::system_error("create input sync object"));
        if (drmSyncobjImportSyncFile(context->drm.get(), work->in_sync, dependency.get()) != 0)
            throw Rejected(detail::system_error("import CSD dependencies"));
    }
    if (drmSyncobjCreate(context->drm.get(), 0, &work->out_sync) != 0)
        throw Rejected(detail::system_error("create output sync object"));
    auto point = std::make_shared<detail::Point>();
    point->context = context;
    work->point = point;
    detail::Prefix successor{point};
    drm_v3d_submit_csd submit{};
    // Physical launch geometry is explicit; it is not inferred from logical
    // tensor shape. Prepared implementations independently own their kernel ABI.
    std::copy(cfg.begin(), cfg.end(), submit.cfg);
    submit.cfg[5] = program->code_->gpu_address();
    submit.cfg[6] = uniform_storage ? uniform_storage->gpu : 0;
    submit.bo_handles = reinterpret_cast<std::uintptr_t>(handles.data());
    submit.bo_handle_count = static_cast<std::uint32_t>(handles.size());
    submit.in_sync = work->in_sync;
    submit.out_sync = work->out_sync;
    state_->pending.push_back(work);
    std::atomic_thread_fence(std::memory_order_release);
    if (::ioctl(context->drm.get(), DRM_IOCTL_V3D_SUBMIT_CSD, &submit) != 0)
        state_->poison("CSD submission failed with potentially partial acceptance");
    int exported = -1;
    if (drmSyncobjExportSyncFile(context->drm.get(), work->out_sync, &exported) != 0 || exported < 0)
        state_->poison("accepted CSD work could not export its final cache-clean completion");
    point->sync_file = detail::Fd(exported);
    point->receipt_valid = true;
    for (const auto& allocation : work->allocations) allocation->ownership.publish(point);
    state_->stream(id).swap(successor);
}

bool Runtime::poisoned() const noexcept {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->fatal;
}

std::string Runtime::failure_reason() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->reason;
}

void Runtime::close() {
    std::unique_lock<std::mutex> lock(state_->mutex);
    if (state_->drained) return;
    state_->healthy(true);
    state_->check_reset();
    detail::Prefix prefix = state_->streams.snapshot_all();
    for (const auto& use : state_->held)
        prefix.insert(prefix.end(), use.prefix.begin(), use.prefix.end());
    // Stop new submissions after constructing the snapshot and before releasing
    // the lock in completion waits. A failed drain does not become success on
    // a later close call merely because submissions have been stopped.
    state_->closed = true;
    state_->wait_prefix(prefix, lock);
    state_->collect_locked();
    if (!state_->pending.empty() || !state_->held.empty()) state_->poison("shutdown failed to settle every tracked native use");
    state_->streams.clear();
    state_->drained = true;
}

} // namespace mstar::pi
