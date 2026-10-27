# The managed native-program boundary

## 1. Why this extension exists

Stage 1 established the native allocation/stream/event substrate and its real
retained Torch consumers on the Pi. A mapped address alone did not let an actual
ggml context consume that storage or join its work into the Torch stream. This
milestone adds that producer–consumer boundary, rather than another runtime.

A submitted implementation still runs within the current Torch stream selected
by its caller. Nothing here changes the philosophical acceleration definition
or asks a model author to define physical buffer imports.

## 2. Owners and bindings

The Pi runtime creates each canonical DRM allocation. Torch Storage retains
that allocation. A provider gets a checked `View`, containing its owner, byte
offset, and extent, through the Torch adapter's supported C++ binding. It cannot
recover ownership by treating `Tensor.data_ptr()` as a portable native handle.

The provider asks the native runtime for the selected DRM major/minor and a
DMA-BUF export of that allocation. It imports the actual BO into ggml's own
Vulkan device, selected by the same physical identity. The imported ggml buffer
is a view of the existing physical storage, not a second model-weight copy.
The import checks the actual allocation size, Vulkan requirements, compatible
memory types, and descriptor alignment/range. Tensor offsets remain relative
to the canonical allocation, including when two tensors alias the same BO.

Managed slots on the same actual ggml VkDevice share a weak DMA-BUF import
cache, keyed by that device and the exported BO's device/inode identity. Every
cache hit checks the canonical logical/backing extents. The cache retains no
buffer, native allocation or device: slot bindings and in-flight owners retain
the actual buffer and native View. Different VkDevices never share Vulkan
handles. This avoids charging an identical prepared weight import to the same
Vulkan heap repeatedly without changing storage ownership or dependencies.

Optional `GGML_VK_MEMORY_LOGGER` accounting follows the final shared buffer
reference, including references retained by another slot or an in-flight use.
Releasing one backend wrapper does not decrement a live shared import. These
diagnostic totals describe logged buffer extents, not measured physical heap
residency or a performance result.

The runtime's normal Vulkan transfer device and ggml's Vulkan device are
separate logical contexts. Their storage identity and explicit fence/foreign
ownership protocol establish interoperability. No code assumes that a fence
or buffer handle from one VkDevice is valid in the other.

`PreparedLinear` owns immutable weight bindings and prepared invocation slots.
Each slot has its own mutable ggml graph/command state. A slot is leased to a
submission until checked retirement. An old Torch Event retains the immutable
completion tail, not the reusable slot. Slot exhaustion rejects before another
submission; it is a readiness/capacity condition for the later runtime consumer.
The binding does not secretly block or grow an unbounded queue to conceal it.

The managed ggml API internally serializes shared host metadata management:
creation, import, preparation, recording/submission, retirement, and inactive
destruction. It does not rely on the Python GIL as its ownership mechanism.
The lock is never held across a GPU completion wait; submitted work can remain
in flight after the call returns. Independent tail query/wait/destruction does
not acquire this metadata lock. Concurrent unmanaged ggml consumers on the same
logical ggml device are outside this initial API's supported coexistence domain.

## 3. Submission has one acceptance boundary

The native `ProviderSubmission` interface separates preparation, possible GPU
acceptance, completion publication, and retirement.

1. Before entering submission, the implementation owns its graph, imports,
   prepared commands/pipelines, and any workspace. It resolves the actual
   current Torch stream and checked input/output views.
2. Under the native ordering lock, the runtime validates declared uses, joins
   the stream prefix and canonical allocation frontiers, and installs Pending.
   Every object needed for retention and publication exists before the provider
   can accept work.
3. The provider receives a borrowed incoming sync fd. It duplicates the fd for
   Vulkan's consuming import. Its acceptance marker is armed immediately before
   the first API call that could accept GPU work.
4. A successful receipt contains an owning exported sync fd and an independent
   completion object that owns the actual provider device/fence. `-1` is allowed
   only for Vulkan's already-signaled payload; it never substitutes for a
   missing completion object.
5. The native runtime publishes this tail to the current stream and every
   declared allocation frontier before returning. An immediate Torch clone or
   resource operation therefore observes the dependency without a CPU wait.

The provider must not reenter Runtime, allocate Torch tensors, or wait for GPU
completion while its submit callback runs under the ordering lock. Prepare
imports/owners outside that callback. Later resource cleanup similarly must
not reenter Runtime from destructors called under its locks.

## 4. The linear and complete graph consumers

The retained Stage 2 `PreparedLinear` consumer admits one 2-D contiguous F32
`MUL_MAT`, with F32 inputs, accumulation, and output. It binds the standard layout:
`weight[M, K]`, `input[N, K]`, and `output[N, M]`.

`PreparedGraph` additionally binds actual multi-operation ggml graphs, with
runtime-owned state, map/mask inputs, canonical intermediate arenas and explicit
output destinations. Its exact operator surface, preparation policy and slot
lease API are described in `managed_graph_provider.md`. Preparation validates
that surface and compiles the actual selected pipelines before serving; the
invocation changes only storage bindings. It uses no additional graph language.

It prepares the genuine ggml F32 shader pipeline, records the graph through
`ggml_backend_graph_compute_async`, and captures ggml's recorded command batches
without submitting them during recording. One checked queue submission contains
foreign acquire, all admitted graph command buffers, and foreign release, and
signals an independent tail. The provider exports that tail and returns it to
the native runtime. No intermediate host completion wait is required.

The one-column (`N=1`) case uses the existing resident F32 vector shader;
multiple columns use the F32 matrix shader. Preparation compiles the same
selected pipeline that recording later uses. Both paths exclude split-K/internal
conversion and require no ggml scratch allocation. The vector shader changes the
F32 reduction order, so target numerical qualification remains required.

The managed path rejects untracked transfer queues, indirect work, deferred
host copies/memsets, unbound tensors, unsupported layouts/dtypes/graphs, and
unsafe output aliasing before acceptance. The CPU ggml backend is not linked.
Both consumers expose prepared, admitted programs. Adding an operation with new
scratch, queue or host-work requirements must extend its real preparation and
completion consumer; accepting a graph name alone would not provide that support.

## 5. Readiness, retirement, and failure

The native runtime retains original prerequisite Points as well as the merged
sync fd. A positive aggregate fence does not erase the need to inspect the
original status/health. It checks both native completion and the provider's
actual completion object, then rechecks device health before releasing owners.

Mutable ggml command state is reset through `ProviderSubmission::retire()` only
after successful completion. Reset errors are reported explicitly; they cannot
be swallowed in a destructor. Failed retirement poisons the runtime and keeps
the Pending owners. Successful retirement returns the slot to its pool even if
an old recorded Event still exists.

An ordinary unsupported/preparation failure before acceptance is a rejection.
Earlier accepted work still belongs to its original drain path. A provider
reports device loss as a typed fatal condition, including when discovered
during preparation/import outside its submit callback. Any exception after
possible acceptance, failed completion export, invalid receipt, completion
error, reset, or uncertain timeout stops admission and retains owners. The
user-visible health authority remains the native runtime.

Ordinary host exceptions after a successful submit do not undo GPU work. The
normal hardware tests exercise an actual drain before cleanup. The separately
gated malformed-receipt test keeps the real receipt privately but deliberately
returns an invalid outward one; it must produce fatal containment, not success
or an automatic fallback. It requires the established controlled reboot path.
This test does not induce a GPU hang and does not qualify the driver's reset,
failed-child-fence, cancellation, or KASAN lifetime behavior.

## 6. Results remain ordinary owned Torch values

The primitive supplies or returns ordinary Torch tensor storage whose owner is
the native allocation. A result is not forced to live in reusable scratch. This
first provider writes into a caller-supplied output, and tests retain that output
and its clone across subsequent invocations.

Deleting Python references while a submission is pending is safe because the
native Pending retains checked allocation owners. Reusing an output as mutable
workspace while an external result still owns it is not authorized by completion.
That is a workspace/resource lease question, separate from storage deallocation.

## 7. Natural next extensions

More operations add admitted graph/program descriptions, preparation, exact
storage uses, and consumers in this same managed provider. Scratch, conversion,
and multiple queues require real preparation and a tail covering all accepted
work. Host continuations require an explicit completion representation; the
current resident-only path must not quietly acquire them.

A future native operator can consume the same supported Torch binding and
`ProviderSubmission` interface without using ggml. Multiple hardware devices
require corresponding placement/storage/event consumers, rather than using
another device's handles. None of those extensions changes the meaning of the
current checked view, stream prefix, returned Tensor, or acceptance boundary.

The subsequent Whisper/M* work binds logical weights and model inputs, M*'s
KV/position/sampling plans, and selected physical programs. This package owns
physical execution. It does not acquire request-state, sampler, deployment,
or model-semantics ownership merely because it has a resident graph.
