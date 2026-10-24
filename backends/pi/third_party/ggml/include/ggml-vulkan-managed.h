#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// External-storage, externally ordered execution. This is a deliberately bounded
// provider surface, not an alternative ggml scheduler or a tensor CPU fallback.
typedef struct ggml_vk_managed * ggml_vk_managed_t;
typedef struct ggml_vk_managed_tail * ggml_vk_managed_tail_t;
typedef void (*ggml_vk_managed_before_accept)(void * user);

// All fallible entry points report errors through this thread-local string.
// Integer status: 0 success, 1 not ready/timeout (tail only), -1 error.
GGML_BACKEND_API const char * ggml_vk_managed_last_error(void);
// True for a device-loss result or uncertain allocation failure (which could
// mask construction of a Vulkan-Hpp device-loss diagnostic). The consumer must
// poison its backend even during preparation/import before the new submission.
GGML_BACKEND_API bool ggml_vk_managed_error_is_fatal(void);

// Selects the physical device by its actual DRM render major/minor. The ggml
// logical device remains ggml-owned. Each handle's lifecycle is externally
// serialized; distinct handles have distinct command/descriptor state and may
// be in flight together. Shared ggml host metadata operations are internally
// serialized, without GPU waits. Managed methods must not be reentered by a
// callback. Concurrent ordinary/unmanaged ggml use of the shared device is not
// part of this provider's supported surface.
GGML_BACKEND_API ggml_vk_managed_t ggml_vk_managed_create(uint32_t drm_major, uint32_t drm_minor);
// Borrowed inspection handle. Do not free it or call ordinary ggml execution,
// synchronization, or transfer entry points on it; use this managed interface.
GGML_BACKEND_API ggml_backend_t ggml_vk_managed_backend(ggml_vk_managed_t managed);
// Fails, without destroying anything, if an invocation has not been retired.
GGML_BACKEND_API int ggml_vk_managed_destroy(ggml_vk_managed_t managed);

// fd is borrowed. A duplicate is imported by Vulkan. The buffer wraps the whole
// canonical BO; tensor views bind offsets within logical_bytes. backing_bytes is
// the real allocation size, independently checked against fd and Vulkan facts.
// The caller owns the returned ggml buffer and must retain it through retirement.
// No mapped-pointer import, host get/set, or implicit transfer is provided.
// The initial matrix program requires each tensor's byte offset to be a multiple
// of the physical device's minStorageBufferOffsetAlignment (32 on the observed
// Pi 5). This is checked, not repaired by an implicit copy.
GGML_BACKEND_API ggml_backend_buffer_t ggml_vk_managed_import_dma_buf(
        ggml_vk_managed_t managed, int fd, uint64_t logical_bytes, uint64_t backing_bytes);

// Shape-only preparation of an immutable actual ggml graph. Storage may remain
// unbound. The admitted F32 operator set covers matmul, im2col, normalization,
// GELU_ERF, softmax, elementwise operations, views/copies and indexed row state.
// I32/I64 conversions use 32-bit words (no shaderInt64 requirement). Matmul uses
// F32 storage/accumulation, with split-K/conversion disabled; softmax width is
// bounded at 16384. These kernels require zero backend-only scratch. The caller
// supplies all graph intermediates as canonical imported storage. Every selected
// pipeline is prepared here. Extending this set requires extending this consumer.
// The graph metadata must not be mutated after preparation; tensor bindings may.
GGML_BACKEND_API int ggml_vk_managed_prepare(ggml_vk_managed_t managed, struct ggml_cgraph * graph);

// All tensors must now be bound to managed imports. Genuine ggml graph execution
// records commands before any GPU submission. The supplied incoming sync_file
// is borrowed (-1 means already satisfied). Foreign acquire, every graph batch,
// and foreign release are submitted on the one admitted compute queue. Transfer
// queues, host continuations, indirect dispatch, and unaccounted work reject.
//
// before_accept MUST be noexcept and mark the caller's submission scope before
// the first possibly accepted vkQueueSubmit. On success, *out_fd is an owned
// sync_file (-1 can mean already signaled), and *out_tail is independent from
// the mutable graph context. On a post-acceptance error, *out_tail is still set;
// retain it, the managed context, graph, and all imports under failure containment.
// A failure before the callback leaves *out_tail=nullptr and *out_fd=-1.
GGML_BACKEND_API int ggml_vk_managed_submit(
        ggml_vk_managed_t managed, struct ggml_cgraph * graph, int dependency_fd,
        ggml_vk_managed_before_accept before_accept, void * user,
        ggml_vk_managed_tail_t * out_tail, int * out_fd);

// Must be called only after the runtime establishes successful completion and
// native health. Checks the independent tail again, then resets reusable graph
// resources. This does not invalidate old event/tail snapshots.
GGML_BACKEND_API int ggml_vk_managed_retire(ggml_vk_managed_t managed);
GGML_BACKEND_API int ggml_vk_managed_tail_query(ggml_vk_managed_tail_t tail);
GGML_BACKEND_API int ggml_vk_managed_tail_wait(ggml_vk_managed_tail_t tail, uint64_t timeout_ns);
// Does not wait or query. Refuses destruction until query/wait/retire has
// observed successful physical completion; inactive teardown is nonfallible.
GGML_BACKEND_API int ggml_vk_managed_tail_destroy(ggml_vk_managed_tail_t tail);

#ifdef __cplusplus
}
#endif
