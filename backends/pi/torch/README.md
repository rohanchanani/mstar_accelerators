# Torch adapter and resident ggml provider

This package exposes the native runtime through Torch 2.9.1's `PrivateUse1`
device, stream/event, storage, and operator interfaces. It is a reusable backend,
not a model implementation or a replacement worker. The initial operator surface
is deliberately explicit; unsupported operations have no blanket CPU fallback.

## Build and process startup

Build the root CMake project first, including its native library, actual ggml
Vulkan provider, and shader targets. `setup.py` consumes the four PIC static
archives from `../build/lib`. Copy `build/shaders/transfer.spv` into
`torch/mstar_pi/transfer.spv` before packaging; the build verifies that it exists.
The delivery's `CODEX_RECIPE.md` gives the complete target build commands.

Install this package with the worker's exact Torch environment:

```sh
python -m pip install --no-build-isolation --no-deps ./torch
```

`MSTAR_PI_BUILD_DIR` can select an isolated root CMake output directory. It
changes only the build input, not runtime discovery. The extension links its
static archive group before the Vulkan and DRM system libraries. `setup.py`
pins Torch 2.9.1 and derives its include paths, libraries, Python ABI and C++ ABI
from that installed Torch.

`python torch/tests/host_preflight.py` checks actual extension import, dispatcher
signatures, guard registration, an unrecorded event, pre-initialization rejection
and a spawned-process repeat. It creates no native context or GPU work and is
safe on the build host. Passing it is a build/registration result, not a hardware
execution claim. The package must be installed (or its parent on `PYTHONPATH`).

Each process imports the package before constructing Pi device objects:

```python
import torch
import mstar_pi

assert mstar_pi.probe("/dev/dri/renderD128")
mstar_pi.initialize(copy_shader="/absolute/path/to/transfer.spv")
with torch.inference_mode():
    x = torch.arange(64, dtype=torch.int32).to("mstar_pi:0")
    y = x.clone()
    assert torch.equal(y.cpu(), torch.arange(64, dtype=torch.int32))
del x, y
mstar_pi.shutdown()
```

Discovery uses a read-only native capability probe; it does not create Vulkan
queues or submit work. Initialization additionally validates the actual provider
and opens the persistent context. It is explicit so a spawned worker receives a
serializable target description and creates its own live resources.

Use `spawn`, never an inherited native context from `fork`. After shutdown or a
fatal context error, start a fresh worker for a new context epoch. Reinitializing
would otherwise allow old Torch stream IDs to refer to new native streams.

## What the first tensor surface supports

- Device storage: FP32, int32 and int64.
- `empty` and `empty_strided`; metadata-only slice/view/reshape paths use Torch's
  existing TensorImpl helpers.
- Equal-shape, equal-dtype, contiguous copy and clone. Copies preserve bits,
  including FP32 and int64; that does not imply native floating or int64 math.
- FP32/int32 fill; int64 fill only when both 32-bit halves repeat (including zero
  and minus one). Other int64 fills reject explicitly.
- Scalar CPU observation through a real completed transfer.
- Generic `torch.Stream` and `torch.Event`, including dependency joins,
  re-recorded-event snapshots, query, synchronization and `record_stream`.
- `torch.mstar_pi.Stream`, `Event` and `stream` facades use those same generic
  Torch objects. They meet M*'s existing backend discovery convention without
  creating a second stream/event system.
- CPU pinned staging backed by owned native allocations, and truthful pointer
  recognition for that allocator.

Noncontiguous allocation/view metadata can exist, but this first copy/fill
provider rejects noncontiguous computation before submission. Dtype conversion,
broadcasting, timing events, graph capture, random operators, autograd and
cross-accelerator copies require later explicit providers. A native whole-node
program can already consume its declared physical views without implementing
every mathematical operation as an ATen operator.

## Transfers, ownership and results

A Storage DataPtr owns a native Allocation plus its context. The mapped CPU
address carried in that DataPtr is opaque when its Tensor is on `mstar_pi`.
Native programs resolve the owned allocation plus tensor offset; they never use
that CPU address as a GPU virtual address.

Every native submission retains declared allocations through completion.
`record_stream` additionally records streams on the Storage owner, then snapshots
their prefixes at the final Storage release. This protects operations enqueued
after the record call and before release. A failed retirement keeps the owner;
it cannot establish safe free by throwing an exception.

A failure in that bookkeeping stops admission and links the owner into an
allocation-free hold list. `health()` reports `retirement_failures` and the
current `retained_storage_owners` separately from native device poison. Successful
shutdown drains accepted work before releasing these holds; failed shutdown
retains them. The failure count remains visible after cleanup.

The provider also orders its physical ownership transfers for a shared backing
allocation. Concurrent read-only consumers on separate Torch streams remain a
valid use: if the Vulkan/direct-CSD ownership mechanism needs serialization,
the native provider inserts that dependency without a host wait. This does not
infer application dataflow or make unordered conflicting writes correct.

Transfer behavior is explicit:

- Pageable CPU to Pi snapshots the source into fresh native staging, then
  enqueues the native copy. Nonblocking submission does not retain a naked CPU
  pointer for a later host read.
- Backend-pinned CPU to Pi copies directly from the native allocation. This can
  consume an earlier device write through a stream dependency without an
  intervening CPU read.
- Pi to backend-pinned CPU enqueues a native write into that allocation. The CPU
  consumer must wait on the stream/event before reading it.
- Pi to pageable CPU waits at the CPU-read boundary, then copies from completed
  staging. It does not return pending writes in untracked ordinary CPU storage.

None of these ownership protections permits overwriting a live subrange. A
workspace/result lease still determines when reuse is allowed. Returning a view
alone cannot transfer that permission. The first retained-result tests use fresh
owned tensors and an owned CPU snapshot.

## Prepared native programs

```python
program = mstar_pi.prepare_csd(machine_words)
program.submit(
    [iteration_count, mstar_pi.address(input_tensor), mstar_pi.address(output_tensor)],
    uses=[],
)
```

An integer uniform is a uint32 immediate. `BufferAddress` is a checked,
allocation-owning address operand; it cannot import an arbitrary CPU pointer.
`uses` declares additional indirectly referenced tensors. The native program's
actual bounds, math and supported domain remain implementation responsibilities.
This first CSD launch form is one 16-lane workgroup; richer launches require a
corresponding native consumer extension.

Preparation uploads code once. Each submission uses the current Torch stream,
consumes its existing prerequisites, and joins native completion before returning
to any retained Torch consumer. No explicit user-level event is required between
a native producer and `tensor.clone()` on that stream.

## Errors and hardware tests

`mstar_pi.health()` is the authoritative context classification after an error.
Direct Python native calls expose `Rejected` and `FatalError`, but Torch can
translate native C++ exceptions from ATen/stream paths into `RuntimeError`.
Workers must not classify device state by parsing exception text.

Closing first stops admission. If a host-side exception interrupts close, the
retained context remains visible as `initialized=True, closed=True` and device-
wide `synchronize()` is still available to establish settlement. Shutdown can
be retried. A successful close releases the registry's context owner; a fatal
close keeps its poisoned status and uncertain owners. Admission closure alone
is never reported as completed execution.

The matching native completion ABI and kernel repair are required by the native
runtime. A signaled fence is not universally proof of safe retirement after an
error. Do not bypass that requirement to obtain a green test.

`tests/consumer_chain.py` performs the actual CSD -> Torch clone -> CSD -> pinned
CPU -> final event chain, plus state, views, output lifetime, storage retirement,
event joins/snapshots, unsupported-operation rejection, exception drain, and a
spawned-worker repeat. Run it under the root supervisor/exclusive device lease.
The supplied frozen program fixture is matched to r2's executed machine code.

The hardware runner reports uncertainty and remains alive on fatal native state.
It does not call timeout, an exception, process exit or `SIGKILL` a drain. Normal
logical test failures drain accepted work before returning failure. No hazardous
hardware fault is intentionally induced.

This test qualifies the native/Torch consumer boundary. It is not a complete M*
worker, a Whisper node, a physical-overlap measurement, or a speedup benchmark.

## Stage 2: actual ggml provider

The supported internal C++ bridge is `csrc/native_binding.h`: `current_execution`
returns this extension's existing native Runtime and current stream;
`tensor_view` returns a checked canonical owner/offset/extent. Providers do not
create a second backend registry or infer storage from a raw tensor pointer.

The first bound program is:

```python
program = mstar_pi.prepare_linear(weights, token_counts=[1, 16], slots=2)
program.submit(inputs, output)
```

`weights` is FP32 `[M, K]`, `inputs` is `[N, K]`, and caller-owned `output` is
`[N, M]`. Preparation selects actual ggml full-FP32 shader pipelines and
prepares bounded per-shape mutable slots. Submission returns after enqueuing
and publishing its tail, without a host completion wait. A following ordinary
Torch clone/staging operation consumes that stream dependency. `info()` reports
the supported shapes, descriptor alignment, and slot/submission/retirement
facts. It is not a measurement of physical overlap or performance.

`ggml_host_preflight.py` exercises registration without implicit initialization.
`ggml_consumer.py` is the actual parent/spawn hardware test; the old
`consumer_chain.py` is frozen as a regression. `ggml_missing_receipt.py` is an
explicitly gated, intentionally held containment test requiring supervised
recovery. Use the root handoff for execution and classification.

This provider does not register arbitrary `aten::mm` or CPU fallback. The
implementation explicitly chooses an admitted prepared program. Its results
are ordinary owned Torch storage, and its native uses are retained independently
of the Python tensor/program objects. New operations belong in the same
preparation/submission consumer, with corresponding supported-domain checks.
