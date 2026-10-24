# Pi native execution provider

This directory implements native execution beneath the Pi Torch backend.
It is C++17, uses Linux DRM and Vulkan C APIs, and has no model or M* imports.
The public boundaries are `include/mstar_pi/runtime.h` and
`include/mstar_pi/provider.h`.

## Supported execution path

- One process-local V3D device context selected by a DRM render-node path.
- A persistent Vulkan context matched to that same physical device using
  `VkPhysicalDeviceDrmPropertiesEXT`; the first enumerated GPU is not assumed.
- DRM-created BO ownership, a CPU mapping, and one persistent canonical Vulkan
  buffer import per allocation. Tensor subviews supply offsets and lengths.
- Four-byte-aligned contiguous bitwise copies and repeating 32-bit fills through
  the included Vulkan compute shader. FP32/int32/int64 copies preserve bits.
- Prepared CSD code, typed physical `CsdLaunch` geometry, and per-submission
  uniform/storage bindings. The current launch validator accepts one 16-lane
  workgroup, supergroup packing of sixteen workgroups, and one batch, matching
  the bounded r2 witness. It rejects other geometry before submission. Adding
  geometry extends the resolver, without changing submission or dependencies.
  The code does not infer geometry from a tensor shape or validate arbitrary
  machine code. Kernel authors remain responsible for the kernel ABI and bounds.
- Multiple logical streams, immutable event snapshots, dependency fan-in,
  native completion queries/waits, and tracked owner retirement.

There is one Vulkan queue and a separate direct-CSD provider. A stream is an
ordered prefix of completion points. `wait_event` joins the event's entire
prefix with the destination's prior prefix. A submitted operation consumes
that join and publishes its new completion. This supports independent physical
providers without making their global host call order an ordering promise.

`record` does not submit artificial GPU work. It snapshots the stream's actual
prefix. Re-recording a Torch Event replaces the outer event's snapshot while
previous waits keep the old one. Prefixes are combined using actual sync_file
fences. An empty DRM sync object is never used as a future execution gate.

The host waiter releases the stream-state mutex during native waits, allowing
another host thread to submit independent work. Accepted work is never canceled
by timeout. A timeout poisons the context and retains its owners.

## Allocation and visibility

The provider creates the intended Vulkan buffer first and queries its actual
memory requirements, including read-ahead padding and dedicated requirements.
It then allocates the DRM backing and imports it into Vulkan. Logical size,
backing size, CPU address, and GPU address remain separate.

The current shared-allocation provider requires a compatible host-visible,
host-coherent memory type. It supplies CPU-write release ordering, Vulkan
HOST/COMPUTE barriers, foreign-queue ownership acquire/release, native dependency
imports, and a final cache-clean fence for CSD. This is not arbitrary host-pointer
import and is not a claim that a CPU access to a pending result is safe.

Vulkan foreign-queue ownership barriers currently cover each entire allocation,
even when a command uses a subview. The provider maintains one physical last-use
frontier for each canonical allocation and joins it into each new use's device
dependencies. This conservatively serializes concurrent readers and disjoint
subranges without host waiting; callers need not invent data dependencies for
the provider's ownership constraint. Real logical read/write dependencies remain
the caller's responsibility. Read/range-aware ownership can later refine this
internal mechanism without changing the stream or primitive interface.

`mapped()` intentionally does not wait. A caller may write a fresh allocation
before submission, or access a previously used region after establishing the
required ordering/completion. The Torch adapter uses this for fresh pageable
input snapshots and for genuinely backend-owned pinned staging allocations.

Native `Pending` objects retain code, uniform, input, output, descriptor,
semaphore, and command-buffer owners until completion is successful. Storage
ownership does not grant permission to overwrite a live subrange. The caller's
resource/workspace lease remains responsible for that permission.

## Driver admission and failure containment

The runtime requires the reviewed private kernel fork ABI in `../kernel`:

- `GET_PARAM 0x50490001` must return exactly completion-contract version `1`.
- `GET_PARAM 0x50490002` must return `0` (healthy). Positive errno is sticky
  device failure until the kernel-defined reboot recovery.

A stock kernel rejects the capability ID, so `Runtime::open` refuses execution.
There is no environment override or kernel-version-string bypass. Mere source
availability is not evidence that the patch is installed or hardware-qualified.

The native machine-code provider also admits only V3D 7.1 revision 6 or later,
using the DRM identity registers. Earlier hardware encodes the CSD batch field
differently and is not covered by this first provider. The register meanings and
revision distinction were checked in the packet's Mesa `v3d_device_info.c`,
`v3d_csd.h`, and `v3dv_cmd_buffer.c`; a Vulkan device-name string is not the gate.

Every completion also checks reset epochs and sync_file execution status.
A native error, unknown submission outcome, missing receipt, changed reset
epoch, or completion timeout poisons this runtime. No fallback or retry follows
an uncertain state transition. `FatalError` means the caller must stop using the
context and its affected state; `Rejected` is an operation rejected before new
native acceptance.

Retirement first inspects every candidate's receipts, then checks kernel health
and reset epochs again, and only then releases qualified owners. A late kernel
error must not be hidden by an already-positive fence. Failure to obtain the
health information also latches containment, including host allocation failure
while constructing an ioctl diagnostic.

On fatal failure the native State retains itself and all tracked owners without
allocating a quarantine list. It deliberately does not destroy potentially live
Vulkan objects or return storage to an allocator. The kernel patch additionally
retains hardware-referenced objects across process termination. Userspace
retention alone would not establish that guarantee.

`close()` stops new submissions, drains the accepted prefix, retires owners, and
marks the runtime drained. Closed and drained are distinct: a failed close
cannot appear to succeed merely because admission was stopped. Existing
Allocations/Events keep their low-level Context alive after a successful close.
The destructor never throws; an unsuccessful destructor drain retains the
context and reports the problem. Serving code must call explicit shutdown so
that a failure is handled, rather than relying on a destructor for reporting.

Live contexts are process-local. Workers must be spawned and initialize their
own backend. Do not fork a process with an initialized native/Torch context or
send Allocation/Event/PreparedCsd objects through IPC.

## Build and validation

The root build compiles `shaders/transfer.comp` to SPIR-V and links Vulkan,
libdrm, and pthread. Pass the resulting absolute SPIR-V path in `Options`.
The selected native source compiles with C++17 warning checks; host compilation
is not a Pi execution result.

Two host tests exercise actual source without a GPU:

- `tests/stream_prefixes_test.cpp`: joined prerequisites cannot lose destination
  work, recorded snapshots do not change, and snapshots retain their points;
  allocation ownership dependencies augment logical prefixes without becoming
  a global dependency or rewriting earlier snapshots.
- `tests/preflight_test.cpp`: discovery and initialization reject unavailable
  devices and unavailable native completion contracts before submission; the
  physical launch resolver rejects unsupported geometry and hardware versions.

The parent handoff's Pi tests must establish the physical operations: CSD to
Vulkan to CSD, dependent Torch clone/state use, mapped staging, views, storage
retention, spawned workers, and failure rejection/drain. Nothing in these host
tests establishes hardware overlap or speedup.

## Current scope

The Stage 2 provider admits a bounded resident ggml F32 matrix program through
the interface below. There is no whole-model program, general ggml graph
admission, capture backend, timing event, arbitrary dtype conversion, pageable
asynchronous output continuation, VC4 transport, or collective backend here.
These are separate consumer/provider extensions. The allocator and completion
ownership are not specialized to the test kernels, and unsupported operations
are rejected explicitly.

## Stage 2: externally ordered providers

The public `provider.h` adds `ProviderSubmission`, `ProviderReceipt`, and an
independent `ProviderCompletion`. `Runtime::submit_provider` consumes checked
allocation uses and a preowned invocation. It installs retention, joins the
stream/allocation dependencies, and holds the ordering lock across acceptance
and tail publication. No callback may reenter Runtime or wait for GPU work.

The mutable invocation is owned by Pending; the immutable completion belongs
to Point. `retire()` performs checked command-state reset only after successful
tail/prerequisites and the final device health check. Fatal or uncertain work
remains retained. `report_provider_failure()` propagates an external fatal
prepare/import observation into the same State. Inside callbacks use the typed
allocation-free `ProviderFatalError`, avoiding lock reentry and allocation
before containment.

`device_identity()` and `export_dma_buf(view)` expose the actual selected DRM
device and canonical allocation import boundary. Exported fds are owning and
CLOEXEC; the caller closes them. Export alone is neither a GPU dependency nor
permission to dereference another provider's native handles. See
`docs/PROVIDER.md` and the real ggml consumer for the complete protocol.
