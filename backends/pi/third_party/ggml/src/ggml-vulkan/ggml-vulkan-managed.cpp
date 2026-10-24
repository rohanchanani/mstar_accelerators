#include "ggml-vulkan-common.h"
#include "ggml-vulkan-managed.h"

#include <cstdio>
#include <cstring>
#include <atomic>
#include <cstdint>
#include <stdexcept>

#ifdef __linux__
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

thread_local char managed_error[768] = {};
thread_local bool managed_error_fatal = false;

// ggml's instance, lazy pipeline maps and tile selectors are shared by managed
// handles. Serialize host management/recording internally; do not depend on a
// language binding's GIL. This mutex never encloses a GPU completion wait, and
// immutable tails have their own independent query/wait/lifetime surface.
std::mutex managed_metadata_mutex;

template <class F> int checked(F && f) noexcept {
    managed_error[0] = 0;
    managed_error_fatal = false;
    try {
        return f();
    } catch (const vk::DeviceLostError & e) {
        managed_error_fatal = true;
        std::snprintf(managed_error, sizeof(managed_error), "%s", e.what());
    } catch (const std::bad_alloc &) {
        // Vulkan-Hpp constructs exceptions after receiving VkResult. An OOM
        // while formatting DEVICE_LOST must not turn into an ordinary rejection.
        // Without reliable provenance, conservatively contain this context.
        managed_error_fatal = true;
        std::snprintf(managed_error, sizeof(managed_error), "managed provider allocation failure; device status may be unavailable");
    } catch (const std::exception & e) {
        std::snprintf(managed_error, sizeof(managed_error), "%s", e.what());
    } catch (...) {
        std::snprintf(managed_error, sizeof(managed_error), "unknown managed Vulkan failure");
    }
    return -1;
}

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

void check_vk(VkResult result, const char * operation) {
    if (result == VK_ERROR_DEVICE_LOST) {
        managed_error_fatal = true;
        throw vk::DeviceLostError(operation);
    }
    if (result != VK_SUCCESS) {
        char message[192];
        std::snprintf(message, sizeof(message), "%s failed with VkResult %d", operation, int(result));
        throw std::runtime_error(message);
    }
}

struct tail_state {
    vk_device device;
    vk::Fence fence;
    bool possibly_accepted = false;
    std::atomic<bool> success_observed{false};

    ~tail_state() {
        if (fence) device->device.destroyFence(fence);
    }
};

struct invocation {
    vk_device device;
    vk::CommandPool boundary_pool;
    vk::Semaphore incoming;
    vk::Semaphore outgoing;
    std::shared_ptr<tail_state> tail;
    std::vector<vk_buffer> buffers;

    ~invocation() {
        // The managed handle owns this until checked retirement. No destructor
        // waits, calls a host continuation, or interprets failure as cancellation.
        if (boundary_pool) device->device.destroyCommandPool(boundary_pool);
        if (incoming) device->device.destroySemaphore(incoming);
        if (outgoing) device->device.destroySemaphore(outgoing);
    }
};

struct recording {
    ggml_backend_vk_context * ctx;
    std::vector<vk::CommandBuffer> commands;
};

thread_local recording * active_recording = nullptr;

struct recording_scope {
    explicit recording_scope(recording & value) {
        require(!active_recording, "nested managed recording is unsupported");
        active_recording = &value;
    }
    ~recording_scope() { active_recording = nullptr; }
};

struct imported {
    std::weak_ptr<vk_buffer_struct> buffer;
    uint64_t logical = 0, backing = 0;
};

using import_key = std::pair<uint64_t, uint64_t>;

// Managed slots share an actual VkDevice. Import each canonical BO once on
// that device, rather than charging its backing size to Vulkan's heap for
// every prepared slot. These weak entries own neither storage nor the device:
// live backend buffers/flight owners retain both. A live buffer therefore also
// prevents reuse of the device address in its key. All access is under the
// existing metadata mutex, outside completion waits.
using device_import_key = std::pair<std::uintptr_t, import_key>;
std::map<device_import_key, imported> managed_imports;

std::vector<const ggml_tensor *> validate_graph(ggml_backend_vk_context * ctx, ggml_cgraph * graph) {
    require(graph && graph->n_nodes > 0, "managed graph has no computation");
    std::vector<const ggml_tensor *> tensors;
    const auto add = [&](const ggml_tensor * tensor) {
        if (tensor && std::find(tensors.begin(), tensors.end(), tensor) == tensors.end()) tensors.push_back(tensor);
    };
    for (int i = 0; i < graph->n_leafs; ++i) add(graph->leafs[i]);
    for (int i = 0; i < graph->n_nodes; ++i) {
        const auto * node = graph->nodes[i];
        add(node);
        add(node->view_src);
        for (const auto * src : node->src) add(src);
        ggml_vk_managed_prepare_pipeline(ctx, node);
    }
    for (const auto * tensor : tensors) {
        require(tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32 || tensor->type == GGML_TYPE_I64,
                "managed graph admits F32 and explicit I32/I64 index storage");
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            require(tensor->ne[d] > 0 && uint64_t(tensor->ne[d]) <= UINT32_MAX && tensor->nb[d] <= UINT32_MAX,
                    "managed tensor dimensions/strides exceed shader indexing");
        }
        require(uint64_t(ggml_nelements(tensor)) <= UINT32_MAX &&
                ggml_nbytes(tensor) <= ctx->device->properties.limits.maxStorageBufferRange,
                "managed tensor exceeds shader/storage-buffer domain");
    }
    return tensors;
}

struct binding {
    vk_buffer buffer;
    uint64_t offset;
    uint64_t size;
};

binding validate_binding(ggml_backend_vk_context * ctx, const ggml_tensor * t) {
    require(t && t->data && t->buffer && ggml_backend_buffer_is_vk(t->buffer),
            "managed tensor is not bound to a Vulkan buffer");
    auto * bc = static_cast<ggml_backend_vk_buffer_context *>(t->buffer->context);
    require(bc && bc->dev_buffer && bc->dev_buffer->managed_external && bc->dev_buffer->device == ctx->device,
            "managed tensor does not reference an import on this ggml device");
    const auto * base_tensor = t->view_src ? t->view_src : t;
    const uintptr_t encoded = reinterpret_cast<uintptr_t>(base_tensor->data);
    const uintptr_t base = reinterpret_cast<uintptr_t>(vk_ptr_base);
    require(encoded >= base, "managed tensor has an invalid encoded buffer offset");
    const uint64_t offset = uint64_t(encoded - base) + t->view_offs;
    require(offset >= t->view_offs, "managed tensor view offset overflows");
    const uint64_t size = ggml_nbytes(t);
    require(offset <= bc->dev_buffer->size && size <= bc->dev_buffer->size - offset,
            "managed tensor view exceeds its canonical allocation");
    require(offset % ggml_type_size(t->type) == 0,
            "managed view offset is not element aligned");
    return {bc->dev_buffer, offset, size};
}

bool overlaps(const binding & a, const binding & b) {
    return a.buffer == b.buffer && a.offset < b.offset + b.size && b.offset < a.offset + a.size;
}

int query_tail(const std::shared_ptr<tail_state> & tail) {
    require(tail && tail->possibly_accepted, "tail has no accepted submission");
    const auto result = tail->device->device.getFenceStatus(tail->fence);
    if (result == vk::Result::eNotReady) return 1;
    check_vk(VkResult(result), "vkGetFenceStatus(managed)");
    tail->success_observed.store(true, std::memory_order_release);
    return 0;
}

#ifdef __linux__
struct fd_owner {
    int value = -1;
    ~fd_owner() { if (value >= 0) close(value); }
    int release() { const int result = value; value = -1; return result; }
};

int duplicate_fd(int fd) {
    const int result = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    require(result >= 0, "could not duplicate external file descriptor");
    return result;
}
#endif

} // namespace

struct ggml_vk_managed_tail {
    std::shared_ptr<tail_state> state;
};

struct ggml_vk_managed {
    ggml_backend_t backend = nullptr;
    ggml_cgraph * prepared = nullptr;
    std::vector<const ggml_tensor *> tensors;
    bool prepared_valid = false;
    bool broken = false;
    std::unique_ptr<invocation> active;
};

// Called at the actual ggml queue-submission boundary. The bounded managed
// program has one queue and no internal semaphore protocol. Rejecting any new
// path here happens before acceptance and requires a corresponding extension.
bool ggml_vk_managed_record_submission(vk_context & ctx, vk::Fence fence) {
    if (!active_recording) return false;
    require(ctx && ctx->p == &active_recording->ctx->compute_cmd_pool,
            "managed graph attempted another queue");
    require(!fence, "managed graph attempted an internal completion fence");
    require(ctx->in_memcpys.empty() && ctx->out_memcpys.empty() && ctx->memsets.empty(),
            "managed graph contains deferred host work");
    for (const auto & sequence : ctx->seqs) {
        for (const auto & submission : sequence) {
            require(submission.buffer && submission.wait_semaphores.empty() && submission.signal_semaphores.empty(),
                    "managed graph attempted an unaccounted semaphore protocol");
            active_recording->commands.push_back(submission.buffer->buf);
        }
    }
    ctx->seqs.clear();
    return true;
}

bool ggml_vk_managed_is_recording(const ggml_backend_vk_context * ctx) {
    return active_recording && (!ctx || active_recording->ctx == ctx);
}

extern "C" const char * ggml_vk_managed_last_error(void) { return managed_error; }
extern "C" bool ggml_vk_managed_error_is_fatal(void) { return managed_error_fatal; }

extern "C" ggml_vk_managed_t ggml_vk_managed_create(uint32_t drm_major, uint32_t drm_minor) {
    ggml_vk_managed_t result = nullptr;
    checked([&] {
        std::lock_guard<std::mutex> metadata_guard(managed_metadata_mutex);
#if !defined(__linux__)
        GGML_UNUSED(drm_major); GGML_UNUSED(drm_minor);
        throw std::runtime_error("managed DMA-BUF provider requires Linux");
#elif defined(GGML_VULKAN_CHECK_RESULTS) || defined(GGML_VULKAN_RUN_TESTS)
        throw std::runtime_error("managed provider forbids ggml result-check/test execution builds");
#else
        ggml_vk_instance_init();
        require(!vk_perf_logger_enabled && !vk_enable_sync_logger,
                "managed provider forbids performance/query and synchronous logging modes");
        const auto devices = vk_instance.instance.enumeratePhysicalDevices();
        size_t selected = SIZE_MAX;
        for (size_t index = 0; index < vk_instance.device_indices.size(); ++index) {
            const auto physical = devices.at(vk_instance.device_indices[index]);
            const auto extensions = physical.enumerateDeviceExtensionProperties();
            const bool has_drm = std::any_of(extensions.begin(), extensions.end(), [](const auto & e) {
                return strcmp(e.extensionName, VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME) == 0;
            });
            if (!has_drm) continue;
            vk::PhysicalDeviceDrmPropertiesEXT drm;
            vk::PhysicalDeviceProperties2 properties;
            properties.pNext = &drm;
            physical.getProperties2(&properties);
            if (drm.hasRender && uint64_t(drm.renderMajor) == drm_major && uint64_t(drm.renderMinor) == drm_minor) {
                require(selected == SIZE_MAX, "DRM identity matched more than one Vulkan physical device");
                require(properties.properties.deviceType != vk::PhysicalDeviceType::eCpu,
                        "managed provider rejects software Vulkan devices");
                selected = index;
            }
        }
        require(selected != SIZE_MAX, "no enumerated ggml Vulkan device matches the DRM render identity");
        auto managed = std::make_unique<ggml_vk_managed>();
        managed->backend = ggml_backend_vk_init(selected);
        require(managed->backend, "ggml Vulkan backend construction failed");
        auto * ctx = static_cast<ggml_backend_vk_context *>(managed->backend->context);
        // Mark ownership immediately: even failed capability validation must
        // use the inactive, nonwaiting teardown. No graph has been submitted.
        ctx->managed_external = true;
        // Ordinary ggml eagerly reserves and clears RMS scratch even for a
        // pure matmul. This program has no RMS operation or scratch need.
        ctx->prealloc_size_add_rms_partials = 0;
        try {
            require(ctx->device->managed_external_interop,
                    "ggml device lacks DMA-BUF/SYNC_FD/FOREIGN interoperability");
            require(ctx->device->support_async && !ctx->device->serialize_submissions,
                    "managed provider requires asynchronous, unserialized ggml execution");
            // Query the external semaphore contract, not just extension names.
            vk::PhysicalDeviceExternalSemaphoreInfo external_info;
            external_info.handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd;
            const auto external = ctx->device->physical_device.getExternalSemaphoreProperties(external_info);
            require(bool(external.externalSemaphoreFeatures & vk::ExternalSemaphoreFeatureFlagBits::eImportable) &&
                    bool(external.externalSemaphoreFeatures & vk::ExternalSemaphoreFeatureFlagBits::eExportable),
                    "SYNC_FD semaphore import/export is unsupported");
        } catch (...) {
            ggml_backend_free(managed->backend);
            managed->backend = nullptr;
            throw;
        }
        result = managed.release();
#endif
        return 0;
    });
    return result;
}

extern "C" ggml_backend_t ggml_vk_managed_backend(ggml_vk_managed_t managed) {
    return managed ? managed->backend : nullptr;
}

extern "C" int ggml_vk_managed_destroy(ggml_vk_managed_t managed) {
    return checked([&] {
        std::lock_guard<std::mutex> metadata_guard(managed_metadata_mutex);
        if (!managed) return 0;
        require(!managed->active, "managed context still owns an unretired invocation");
        ggml_backend_free(managed->backend);
        managed->backend = nullptr;
        delete managed;
        return 0;
    });
}

extern "C" ggml_backend_buffer_t ggml_vk_managed_import_dma_buf(
        ggml_vk_managed_t managed, int fd, uint64_t logical_bytes, uint64_t backing_bytes) {
    ggml_backend_buffer_t result = nullptr;
    checked([&] {
        std::lock_guard<std::mutex> metadata_guard(managed_metadata_mutex);
#ifndef __linux__
        GGML_UNUSED(managed); GGML_UNUSED(fd); GGML_UNUSED(logical_bytes); GGML_UNUSED(backing_bytes);
        throw std::runtime_error("managed DMA-BUF provider requires Linux");
#else
        require(managed && !managed->broken && !managed->active, "managed context is not available for import");
        for (auto entry = managed_imports.begin(); entry != managed_imports.end();) {
            if (entry->second.buffer.expired()) entry = managed_imports.erase(entry);
            else ++entry;
        }
        auto * ctx = static_cast<ggml_backend_vk_context *>(managed->backend->context);
        const auto & device = ctx->device;
        require(fd >= 0 && logical_bytes && logical_bytes <= backing_bytes && backing_bytes <= SIZE_MAX,
                "invalid canonical DMA-BUF extents");
        require(logical_bytes <= device->max_buffer_size, "canonical import exceeds device buffer-size limit");
        struct stat identity {};
        require(fstat(fd, &identity) == 0, "could not stat canonical DMA-BUF");
        const off_t actual_size = lseek(fd, 0, SEEK_END);
        require(actual_size >= 0 && uint64_t(actual_size) == backing_bytes, "DMA-BUF backing size does not match allocation metadata");
        const device_import_key key{reinterpret_cast<std::uintptr_t>(device.get()),
                                    {uint64_t(identity.st_dev), uint64_t(identity.st_ino)}};
        vk_buffer buffer;
        auto found = managed_imports.find(key);
        if (found != managed_imports.end()) {
            buffer = found->second.buffer.lock();
            if (buffer) require(found->second.logical == logical_bytes && found->second.backing == backing_bytes,
                                "aliased DMA-BUF imports disagree about canonical extents");
        }
        if (!buffer) {
            const vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer;
            vk::PhysicalDeviceExternalBufferInfo external_info;
            external_info.usage = usage;
            external_info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT;
            const auto external = device->physical_device.getExternalBufferProperties(external_info);
            require(bool(external.externalMemoryProperties.externalMemoryFeatures & vk::ExternalMemoryFeatureFlagBits::eImportable),
                    "storage-buffer DMA-BUF import is unsupported");
            vk::ExternalMemoryBufferCreateInfo external_buffer;
            external_buffer.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT;
            vk::BufferCreateInfo buffer_info;
            buffer_info.size = logical_bytes;
            buffer_info.usage = usage;
            buffer_info.sharingMode = vk::SharingMode::eExclusive;
            buffer_info.pNext = &external_buffer;
            auto imported_buffer = std::make_shared<vk_buffer_struct>();
            imported_buffer->device = device;
            // Establish the destructor's owner before any resource is created.
            imported_buffer->size = logical_bytes;
            imported_buffer->managed_external = true;
            imported_buffer->buffer = device->device.createBuffer(buffer_info);
            vk::MemoryDedicatedRequirements dedicated;
            vk::MemoryRequirements2 requirements;
            requirements.pNext = &dedicated;
            vk::BufferMemoryRequirementsInfo2 requirements_info(imported_buffer->buffer);
            device->device.getBufferMemoryRequirements2(&requirements_info, &requirements);
            require(requirements.memoryRequirements.size <= backing_bytes && requirements.memoryRequirements.alignment != 0,
                    "canonical BO cannot back this ggml Vulkan buffer");
            const auto fd_properties = device->device.getMemoryFdPropertiesKHR(vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT, fd);
            const uint32_t types = requirements.memoryRequirements.memoryTypeBits & fd_properties.memoryTypeBits;
            const auto memory = device->physical_device.getMemoryProperties();
            uint32_t memory_type = UINT32_MAX;
            for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
                if ((types & (1u << i)) && memory.memoryHeaps[memory.memoryTypes[i].heapIndex].size >= backing_bytes) {
                    memory_type = i;
                    if (memory.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) break;
                }
            }
            require(memory_type != UINT32_MAX, "DMA-BUF and ggml buffer have no compatible memory type");
            const bool dedicated_only = dedicated.requiresDedicatedAllocation ||
                bool(external.externalMemoryProperties.externalMemoryFeatures & vk::ExternalMemoryFeatureFlagBits::eDedicatedOnly);
            if (dedicated_only) require(backing_bytes == requirements.memoryRequirements.size,
                                        "dedicated import size differs from the canonical BO size");
            fd_owner owned{duplicate_fd(fd)};
            vk::ImportMemoryFdInfoKHR import_info;
            import_info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT;
            import_info.fd = owned.value;
            vk::MemoryDedicatedAllocateInfo dedicated_info;
            dedicated_info.buffer = imported_buffer->buffer;
            if (dedicated_only) import_info.pNext = &dedicated_info;
            vk::MemoryAllocateInfo allocation(backing_bytes, memory_type, &import_info);
            imported_buffer->device_memory = device->device.allocateMemory(allocation);
            owned.release(); // Vulkan owns the descriptor only after success.
            device->device.bindBufferMemory(imported_buffer->buffer, imported_buffer->device_memory, 0);
            imported_buffer->memory_property_flags = memory.memoryTypes[memory_type].propertyFlags;
            device->memory_logger->log_allocation(imported_buffer, logical_bytes);
            buffer = std::move(imported_buffer);
            managed_imports[key] = {buffer, logical_bytes, backing_bytes};
        }
        const auto buft = ggml_backend_vk_buffer_type(device->idx);
        std::string name = "VulkanManagedDMA_BUF";
        auto bc = std::make_unique<ggml_backend_vk_buffer_context>(device, std::move(buffer), name);
        result = ggml_backend_buffer_init(buft, ggml_backend_vk_buffer_interface, bc.get(), logical_bytes);
        require(result, "could not create managed ggml backend buffer");
        bc.release();
#endif
        return 0;
    });
    return result;
}

extern "C" int ggml_vk_managed_prepare(ggml_vk_managed_t managed, ggml_cgraph * graph) {
    return checked([&] {
        std::lock_guard<std::mutex> metadata_guard(managed_metadata_mutex);
        require(managed && !managed->broken && !managed->active, "managed context is not available for preparation");
        auto * ctx = static_cast<ggml_backend_vk_context *>(managed->backend->context);
        const auto next = validate_graph(ctx, graph);
        managed->tensors = next;
        managed->prepared = graph;
        managed->prepared_valid = true;
        return 0;
    });
}

extern "C" int ggml_vk_managed_submit(
        ggml_vk_managed_t managed, ggml_cgraph * graph, int dependency_fd,
        ggml_vk_managed_before_accept before_accept, void * user,
        ggml_vk_managed_tail_t * out_tail, int * out_fd) {
    if (out_tail) *out_tail = nullptr;
    if (out_fd) *out_fd = -1;
    return checked([&] {
        std::lock_guard<std::mutex> metadata_guard(managed_metadata_mutex);
#ifndef __linux__
        GGML_UNUSED(managed); GGML_UNUSED(graph); GGML_UNUSED(dependency_fd);
        GGML_UNUSED(before_accept); GGML_UNUSED(user);
        throw std::runtime_error("managed DMA-BUF provider requires Linux");
#else
        require(managed && !managed->broken && !managed->active && out_tail && out_fd && before_accept,
                "invalid managed submission or occupied context");
        auto * ctx = static_cast<ggml_backend_vk_context *>(managed->backend->context);
        // Graph metadata is immutable after preparation. Only bound storage changes.
        require(managed->prepared_valid && managed->prepared == graph,
                "managed graph identity differs from its prepared program");
        std::vector<binding> bindings;
        bindings.reserve(managed->tensors.size());
        for (const auto * tensor : managed->tensors) bindings.push_back(validate_binding(ctx, tensor));
        for (int i = 0; i < graph->n_nodes; ++i) {
            const auto * node = graph->nodes[i];
            if (node->op == GGML_OP_MUL_MAT) {
                const auto a = validate_binding(ctx, node->src[0]);
                const auto b = validate_binding(ctx, node->src[1]);
                const auto d = validate_binding(ctx, node);
                const auto alignment = ctx->device->properties.limits.minStorageBufferOffsetAlignment;
                require(a.offset % alignment == 0 && b.offset % alignment == 0 && d.offset % alignment == 0,
                        "managed matrix view is not storage-buffer aligned");
                require(!overlaps(d, a) && !overlaps(d, b), "managed matrix output aliases a live input");
            }
        }
        require(dependency_fd >= -1, "invalid incoming dependency descriptor");
        auto resources = std::make_unique<invocation>();
        resources->device = ctx->device;
        resources->tail = std::make_shared<tail_state>();
        resources->tail->device = ctx->device;
        resources->tail->fence = ctx->device->device.createFence({});
        for (const auto & bound : bindings) {
            if (std::find(resources->buffers.begin(), resources->buffers.end(), bound.buffer) == resources->buffers.end()) {
                resources->buffers.push_back(bound.buffer);
            }
        }
        auto tail = std::make_unique<ggml_vk_managed_tail>();
        tail->state = resources->tail;
        managed->active = std::move(resources);
        try {
            auto & execution = *managed->active;
            auto & device = ctx->device;
            recording records{ctx, {}};
            {
                recording_scope scope(records);
                require(ggml_backend_graph_compute_async(managed->backend, graph) == GGML_STATUS_SUCCESS,
                        "ggml graph recording failed");
            }
            require(!records.commands.empty() && ctx->compute_ctx.expired() && ctx->transfer_ctx.expired(),
                    "ggml graph left unsubmitted or unaccounted work");
            require(!ctx->submit_pending && !ctx->almost_ready_fence_pending,
                    "ggml graph requested an internal completion protocol");
            const uint32_t family = device->compute_queue->queue_family_index;
            execution.boundary_pool = device->device.createCommandPool({vk::CommandPoolCreateFlagBits::eTransient, family});
            const auto boundary = device->device.allocateCommandBuffers({execution.boundary_pool, vk::CommandBufferLevel::ePrimary, 2});
            std::vector<vk::BufferMemoryBarrier> acquires, releases;
            for (const auto & buffer : execution.buffers) {
                vk::BufferMemoryBarrier acquire;
                acquire.srcAccessMask = {};
                acquire.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
                acquire.srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
                acquire.dstQueueFamilyIndex = family;
                acquire.buffer = buffer->buffer;
                acquire.offset = 0;
                acquire.size = VK_WHOLE_SIZE;
                acquires.push_back(acquire);
                auto release = acquire;
                release.srcAccessMask = acquire.dstAccessMask;
                release.dstAccessMask = {};
                release.srcQueueFamilyIndex = family;
                release.dstQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
                releases.push_back(release);
            }
            boundary[0].begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            boundary[0].pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader,
                                        {}, {}, acquires, {});
            boundary[0].end();
            boundary[1].begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            boundary[1].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eBottomOfPipe,
                                        {}, {}, releases, {});
            boundary[1].end();
            if (dependency_fd >= 0) {
                execution.incoming = device->device.createSemaphore({});
                fd_owner incoming_fd{duplicate_fd(dependency_fd)};
                vk::ImportSemaphoreFdInfoKHR info;
                info.semaphore = execution.incoming;
                info.flags = vk::SemaphoreImportFlagBits::eTemporary;
                info.handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd;
                info.fd = incoming_fd.value;
                device->device.importSemaphoreFdKHR(info);
                incoming_fd.release();
            }
            vk::ExportSemaphoreCreateInfo export_info;
            export_info.handleTypes = vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd;
            vk::SemaphoreCreateInfo semaphore_info;
            semaphore_info.pNext = &export_info;
            execution.outgoing = device->device.createSemaphore(semaphore_info);
            std::vector<vk::CommandBuffer> commands;
            commands.reserve(records.commands.size() + 2);
            commands.push_back(boundary[0]);
            commands.insert(commands.end(), records.commands.begin(), records.commands.end());
            commands.push_back(boundary[1]);
            const vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eAllCommands;
            vk::SubmitInfo submit;
            if (execution.incoming) {
                submit.waitSemaphoreCount = 1;
                submit.pWaitSemaphores = &execution.incoming;
                submit.pWaitDstStageMask = &wait_stage;
            }
            submit.commandBufferCount = uint32_t(commands.size());
            submit.pCommandBuffers = commands.data();
            submit.signalSemaphoreCount = 1;
            submit.pSignalSemaphores = &execution.outgoing;
            // Every owner, Vulkan object, command, and C tail is established
            // before this edge. No accepted failure executes cleanup below.
            *out_tail = tail.release();
            execution.tail->possibly_accepted = true;
            before_accept(user);
            device->compute_queue->handle->submit({submit}, execution.tail->fence);
            vk::SemaphoreGetFdInfoKHR get_fd;
            get_fd.semaphore = execution.outgoing;
            get_fd.handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd;
            fd_owner exported{device->device.getSemaphoreFdKHR(get_fd)};
            require(exported.value >= -1, "Vulkan returned an invalid completion sync_file");
            if (exported.value >= 0) require(fcntl(exported.value, F_SETFD, FD_CLOEXEC) == 0,
                                            "could not set completion sync_file close-on-exec");
            *out_fd = exported.release();
            return 0;
        } catch (...) {
            if (!managed->active->tail->possibly_accepted) {
                // No VkQueueSubmit occurred. Discard recorded commands before
                // releasing imported buffers. Failure to reset makes this slot
                // unusable; it does not fabricate a successfully retired slot.
                try {
                    ctx->compute_ctx.reset();
                    ctx->transfer_ctx.reset();
                    ctx->submit_pending = false;
                    ctx->almost_ready_fence_pending = false;
                    ggml_vk_graph_cleanup(ctx);
                } catch (const vk::DeviceLostError &) {
                    managed_error_fatal = true;
                    managed->broken = true;
                } catch (const std::bad_alloc &) {
                    managed_error_fatal = true;
                    managed->broken = true;
                } catch (...) {
                    managed->broken = true;
                }
                managed->active.reset();
            }
            throw;
        }
#endif
    });
}

extern "C" int ggml_vk_managed_retire(ggml_vk_managed_t managed) {
    return checked([&] {
        std::lock_guard<std::mutex> metadata_guard(managed_metadata_mutex);
        require(managed && !managed->broken && managed->active, "managed context has no retireable invocation");
        require(query_tail(managed->active->tail) == 0, "managed invocation is still in physical use");
        auto * ctx = static_cast<ggml_backend_vk_context *>(managed->backend->context);
        // The caller has additionally checked its native kernel/reset/status
        // contract. Fence readiness alone never establishes native health.
        ggml_vk_graph_cleanup(ctx);
        managed->active.reset();
        return 0;
    });
}

extern "C" int ggml_vk_managed_tail_query(ggml_vk_managed_tail_t tail) {
    return checked([&] {
        require(tail, "null managed tail");
        return query_tail(tail->state);
    });
}

extern "C" int ggml_vk_managed_tail_wait(ggml_vk_managed_tail_t tail, uint64_t timeout_ns) {
    return checked([&] {
        require(tail && tail->state && tail->state->possibly_accepted, "tail has no accepted submission");
        const auto result = tail->state->device->device.waitForFences({tail->state->fence}, true, timeout_ns);
        if (result == vk::Result::eTimeout) return 1;
        check_vk(VkResult(result), "vkWaitForFences(managed)");
        tail->state->success_observed.store(true, std::memory_order_release);
        return 0;
    });
}

extern "C" int ggml_vk_managed_tail_destroy(ggml_vk_managed_tail_t tail) {
    return checked([&] {
        if (!tail) return 0;
        if (tail->state->possibly_accepted) require(tail->state->success_observed.load(std::memory_order_acquire),
                "managed tail has no success observation; retain it until checked completion");
        delete tail;
        return 0;
    });
}
