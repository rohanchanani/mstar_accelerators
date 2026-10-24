# Native/Torch integration contract

## 1. The owner is a worker-local backend context

`mstar_pi.initialize()` constructs one native runtime in the spawned worker
process. That context owns the selected DRM file, matching Vulkan device/queue,
allocation machinery, prepared transfer pipeline, completion objects, and
retirement state. Importing the Python package registers the backend; it does
not create that context or submit hardware work.

The runtime selects Vulkan by the DRM render device's actual major/minor
identity. It does not choose the first enumerated GPU, accept a software Vulkan
device, or treat a CPU tensor's address as a GPU virtual address. Read-only
discovery and full initialization are separate operations.

One backend instance exposes `mstar_pi:0`. There can be several logical streams
within it. The native transfer context has one Vulkan queue plus the direct-CSD provider.
The Stage 2 ggml provider imports canonical storage into its own actual Vulkan
context and publishes its tail through the same native stream machinery.
Multiple streams or contexts do not imply physical parallelism. Multiple
devices and interprocess execution require corresponding consumers; they are
not advertised by this version.

Workers use spawn and create their own context. An initialized context is not
fork-safe or an IPC object. Shutdown does not silently create a new epoch with
reused stream IDs: a new context requires a new worker process.

## 2. Storage ownership is independent of tensor metadata

The native allocator creates a DRM BO and one canonical Vulkan buffer import.
It queries the intended buffer's actual backing requirements before allocation.
The logical byte extent, backing allocation size, CPU mapping, and GPU address
are distinct facts.

A Torch Storage owns a shared reference to that native allocation. Tensor views
retain the Storage and supply offsets, sizes, and strides. Native operations
accept checked views with explicit owners. A `BufferAddress` binds a Tensor
view to a GPU address and retains the allocation; raw host pointers are not an
import mechanism. Indirectly addressed buffers must also appear in the native
program's declared uses.

A native submission retains code, uniforms, inputs, outputs, and submission
objects until their physical use has completed successfully. Torch
`record_stream` adds allocator-use tracking. Deleting the last Python Tensor
reference cannot free a BO still retained by a submission.

This does not authorize overwriting a live region. Runtime resource/workspace
leases remain responsible for conflicting reuse. A returned Tensor owns its
Storage through normal Torch references; an output that aliases workspace must
not be reused while the result is live. The first hardware test allocates real
output storage and deliberately retains an earlier result across a later call.

The present Vulkan foreign-ownership barriers cover whole allocations. The
native allocator therefore tracks each allocation's prior physical use and
joins that completion into the next operation's device dependencies. This
also handles legal Torch read/read consumers on different streams: callers do
not have to invent a logical data dependency for a physical ownership transfer.
It is conservative device-side serialization, not a host wait. Finer ranges or
provider-aware read sharing can extend this mechanism without changing Tensor
ownership or the stream/event contract. Shared prepared code is conservatively
tracked too; no concurrency or performance claim follows from this first policy.

## 3. A stream is an ordered prefix; an event is a snapshot

Each stream has a prefix of native completion points. A submission consumes the
entire prefix, including all imported prerequisites, and establishes a new
completion point. `wait_event` joins the destination's existing prefix with the
event's snapshot. It cannot replace and thereby forget the destination's prior
work.

Recording an event snapshots the prefix without an artificial GPU operation.
Re-recording that outer Torch Event creates a new snapshot. Earlier waits keep
their old snapshot, including ownership of its completion points.

Cross-provider ordering uses actual sync_file dependencies, imported/exported
through DRM sync objects and Vulkan SYNC_FD semaphores. The native code also
establishes the required visibility and ownership barriers. These dependencies
are inserted before the consuming operation. Waiting only at the worker's
event after an unordered Torch clone would be too late.

CPU consumption is a distinct boundary. A backend-owned pinned CPU Tensor may
be the destination of a nonblocking GPU copy, but CPU code must wait for the
relevant event before reading it. Copying to an ordinary pageable CPU Tensor
uses owned staging and an explicit completion wait before CPU materialization.
The adapter does not promise deferred host copies that its event cannot cover.

An event may already be complete when queried. The tests require correct
ordering and results; they do not require a particular operation to be slow
enough for a pending observation.

## 4. Successful execution and storage settlement are different

There are three consequential outcomes:

| Outcome | Meaning | Required action |
| --- | --- | --- |
| Preflight rejection | This operation accepted no new native work | Report unsupported input/capability; previously submitted work retains its normal obligations |
| Successful completion | All represented work completed with accepted status and visibility | Publish/read results at the appropriate boundary; retire tracked physical uses |
| Fatal or uncertain outcome | Submission may be partial, completion failed, a reset occurred, or a deadline expired | Stop admission; invalidate affected computation/state; retain uncertain owners; recover the device through the defined path |

`Rejected` is not a claim that the whole context is idle. An ordinary host
exception after an earlier submission must still drain that submission before
ordinary cleanup. The hardware test exercises this without deliberately
hanging or resetting the GPU.

`FatalError` poisons the native context. The Torch dispatcher may translate a
C++ exception into `RuntimeError`, so the worker-facing distinction is obtained
from `mstar_pi.health()`, not by parsing exception text. A timeout is not a
cancel, and an error-signaled fence alone does not establish safe storage reuse.

The native State retains itself and all uncertain owners on fatal exits.
Containment is established before fallible diagnostic formatting. Explicit
shutdown stops new submission and drains tracked work; a failed close cannot
subsequently appear successful merely because the context is closed.

The Torch adapter distinguishes closed admission from access to completion.
If shutdown encounters a host error, the retained context can still be drained
with device-wide synchronization. A Storage-deleter bookkeeping failure retains
that owner in an allocation-free list, stops admission, and reports counters
and a reason through health. Successful shutdown releases those held owners
only after the native drain; fatal failure retains them. It cannot silently
turn missing retirement bookkeeping into a healthy, empty allocator.

Userspace retention cannot protect storage after a process is killed. The
matching kernel repair therefore retains uncertain hardware-referenced jobs,
BOs, and associated owners. Its first recovery policy is deliberately a
controlled reboot. It does not claim transparent retry, state recovery, forced
module unload safety, hot removal, or in-place reset recovery.

## 5. The driver capability is a real producer/consumer boundary

This source requires private GET_PARAM `0x50490001 == 1` and healthy status
`0x50490002 == 0`. A stock kernel must reject admission. The private ABI means
the specific supplied completion repair is present; it is not an upstream
standard or a performance/correctness certificate for arbitrary QPU programs.

The repair addresses identified submission ordering, dependency error, cache
clean, cancellation, and reset paths in the pinned source. The code also checks
sync_file status and reset epochs. A passing source test is not a compiled
kernel or a hardware test. The handoff explicitly separates those stages.

## 6. What the implementation author supplies

An implementation supplies its program, physical launch description, uniform
values and typed address operands, and any additional storage uses. Prepared
code persists across invocations. The current direct-CSD launch subset is
explicitly bounded; the backend rejects unsupported physical geometry before
submission. Expanding that subset means extending its encoder and tests, not
introducing kernel-name cases into the runtime.

The implementation author remains responsible for algorithmic correctness,
machine-code bounds, and dependency obligations beyond those expressed through
the supplied execution stream. This is ordinary implementation responsibility,
not a mandatory proof language. The backend is not a sandbox for arbitrary
machine code.

This package's Torch current-stream scope is the concrete execution context at
the boundary. A bound M* primitive receives its runtime-selected execution
context and calls this provider on that stream. It need not manufacture a second
completion system. Additional provider queues must join the context before a
retained consumer uses their results.

## 7. Current consumers and extension boundaries

**ggml provider.** The bounded resident FP32 program and complete managed graphs
are described in `PROVIDER.md` and `managed_graph_provider.md`. Their real Vulkan
context consumes canonical allocations and incoming/outgoing dependencies.
Additional scratch, transfer queues and host continuations still need explicit
consumers; a pointer is not an allocation or completion contract.

**Whisper and M*.** The companion candidate constructs fresh model-owned
encoder/decoder shells and binds logical weights, complete physical programs,
KV/position/sampling plans and outputs. Its generated execution capabilities
carry supported modes/shapes and workspace admission. The worker consumes
backend failure without normal cleanup of uncertain work. Full target gates
remain necessary; host integration is not native qualification.

**Other devices and distributed execution.** CUDA/ROCm retain their Torch
implementations. A VC4 service or another device supplies its storage/submission
consumer under the same ordering and ownership obligations. Distributed roles
still require agreed implementation choice and compatible collective
participation. Nothing here claims that a single-device fence is a distributed
completion object.

## Source grounding

The inspected serving source is original M* commit
`fa37d227c3467145a2fe690cd53924ec9d678f90`, not the provisional Pi integration.

- `mstar/engine/engine.py`, lines 1845–1862: immediate result cloning during
  output merge motivates the native-to-Torch consumer join.
- `mstar/worker/worker.py`, lines 1978–2008: execution stream, staging, and final
  Torch event; lines 2994–3006: event/device wait and CPU fast path.
- Pi r2 `NATIVE_EXECUTION_R2.md` and `KERNEL_CONTRACT.md`: native shared-BO
  execution evidence and source-established completion failures.
- Pi r2 `probes/csd-r2/native-contract-final.stdout`: machine-code hashes used
  by the supplied fixtures.
- `kernel/source_manifest.json`: exact original and patched source identities.

The complete source and raw packets remain the evidence. Stage 1 and Stage 2
passed their normal paths on the real Pi. Those observations do not qualify the
new complete managed-graph consumer or full Whisper deployment.
