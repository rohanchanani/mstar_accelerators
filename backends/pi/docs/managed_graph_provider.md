# Managed ggml graph provider

`mstar_pi.ggml.Graph` is a thin owner of a real `ggml_context`, graph and tensors.
The Python construction calls map directly to ggml operations. They do not define
another intermediate language. Graph metadata can be authored and checked on a
host with no Pi device. `prepare_graph` requires the explicitly initialized Pi
runtime and calls the factory once per independently leased mutable slot.

## Ownership and execution

The native Pi `Runtime` remains the allocation, stream, dependency, health and
retirement authority. Constants, invocation inputs, state, maps, masks, output
storage and each slot's intermediate arena all enter `submit_provider`'s `uses`.
The provider imports their canonical DMA-BUF storage; it neither interprets a
mapped pointer as an import nor allocates a parallel physical weight/state heap.

Preparation binds constants, allocates a canonical intermediate arena, prepares
all selected pipelines, and freezes the graph's metadata. Only storage bindings
change on a later invocation. The first arena policy assigns disjoint aligned
storage to every non-view intermediate. The graph interface and lifetime model
also permit a future liveness-based arena allocator; that optimization must
establish its alias dependencies rather than silently overlap live values.

`try_acquire()` reserves a slot without waiting. The scheduler acquires it before
resource admission. `GraphLease.close()` returns an unsubmitted slot. On submit,
a `GraphFlight` receives the lease, imports and all native allocation views before
any command can be accepted. The runtime's acceptance marker, complete incoming
prefix and immutable tail remain the Stage 2 protocol. Physical completion and
successful health checks permit retirement and slot reuse. Closing the Python
lease after successful submission cannot release the flight. Old Torch events
retain an independent tail even after the graph slot is reused.

`PreparedGraph.close()` is explicit teardown: it drains through the native
runtime's normal synchronization/retirement path, rejects any unreturned lease,
and releases the prepared contexts. It is not called on the serving hot path.
An uncertain failure retains the runtime/flight/imports; it is never a normal
preaccept rejection or permission to release borrowed storage.

The ordinary ggml almost-ready fence is disabled for managed graphs. The provider
publishes one truthful completion scope covering every recorded graph batch.
Managed graphs also disable ordinary ggml fusion: preparation and execution use
the same admitted pipelines. Existing ggml range-based synchronization handles
read/write aliases inside the graph, including `set_rows` followed by `get_rows`.

## Admitted numerical and storage policy

The first full graph consumer supports FP32 im2col, matrix multiplication,
normalization, GELU_ERF, elementwise add/multiply/scale, softmax, row gather/scatter,
contiguous copies, views/permutations/reshapes, argmax and index conversion.
Package 2 adds the optional read-only-seed/offset categorical operation described
in [categorical.md](categorical.md), using the same prepared graph/storage path.
Matrix operands and accumulation stay FP32. Split-K and internal numerical
conversion are disabled. Softmax rows are at most 16,384 elements. This exact
operator policy requires zero backend-only scratch; all graph intermediates are
in the supplied canonical arena. Adding a scratch-requiring path extends
preparation and binding, rather than allowing execution to allocate hidden BOs.

Managed im2col explicitly selects the ordinary storage-buffer shader even on a
device capable of buffer device addresses. Canonical imports do not acquire an
unadvertised device-address allocation contract. Managed norm, GELU, copy,
gather/scatter and softmax use at most 256 invocations per workgroup, respecting
the observed V3D limit; ordinary ggml's 512-invocation variants remain separate.
The normalization shader also checks the final rounded row tile before any
storage access. The I32/I64 conversion shaders
represent I64 storage as two 32-bit words. They require no `shaderInt64` feature.
I64-to-I32 conversion is for the runtime's checked row/token index domain; it does
not claim arbitrary int64 arithmetic. Row indices must be in range and scatter
indices unique. Those facts belong to the bound resource plan.

Argmax reduces `(value, original_index)` pairs and preserves the first index for
ties, all negative infinity, signed zero and NaNs, matching the target Torch
selection contract. The device gate compares those cases directly.

`Graph.allocation_plan(alignment)` reports the exact canonical intermediate
arena plan from real ggml tensor metadata without initializing a device. The
same planner is used by preparation after querying the actual backend alignment.
It excludes weights, persistent state, invocation outputs, staging and driver
allocations; it is an input to a memory budget, not a target-fit measurement.

## Qualification commands

After building and installing this source's native library and Torch extension:

```sh
python torch/tests/host_preflight.py
python torch/tests/ggml_host_preflight.py
python torch/tests/graph_host_preflight.py
python torch/tests/categorical_reference.py
python torch/tests/check_graph_spirv.py build/third_party/ggml/src/ggml-vulkan/vulkan-shaders.spv
```

The first two unchanged Stage 1/Stage 2 hardware gates remain required. Run the
new gate through the same shared hardware admission launcher:

```sh
python tools/run_hardware.py --deadline 600 -- \
  python torch/tests/graph_consumer.py \
    --shader build/shaders/transfer.spv --validation
```

The new gate exercises the real multi-operation graph, writes and rereads runtime
state through returned aliases, checks nonzero-offset views and guards, consumes
results immediately through Torch clone/pinned output/events, retains outputs
across invocation reuse, releases Python owners, checks slot admission and
rejection, and repeats in a spawned worker. CPU computation supplies independent
test oracles only. It is not a provider fallback. Passing this gate is not a full
Whisper accuracy claim, physical-overlap measurement, speedup result, or
exceptional kernel/reset qualification.
