# Ordinary Torch implementations

`mstar_accelerators.backends.torch.runtime:TorchRuntime` binds the ordinary
registry in an explicit CPU deployment. It does not rewrite device predicates,
interpret ggml, import the native Pi backend, or construct reference programs.
M* owns worker processes, communication groups, request lifecycle and shutdown.
CPU Torch callbacks are synchronous. Their collectives use the placement-derived
groups supplied through `NodeBinding`.

The runtime's `threads` setting applies to the worker process. An optional
`node_threads: {audio_encoder: 2}` mapping overrides it once during node assembly,
before weight preparation or execution threads start. Unmapped nodes use
`threads`. Nodes sharing one worker must request the same count; conflicting
counts are rejected during assembly. Use M* `node_groups` placement to give nodes
with different counts separate workers. The runtime logs each node's actual
thread count and never changes this policy during execution.

## Generated calls and synchronous programs

An author can use any `PreparedImplementation` callbacks. The optional
`TorchProgram(run, slots=2)` / `prepare_torch_programs` helper supplies a small
ordinary-Torch realization of that same interface. Pass the generated
`Inputs`, `Plans` and `Call` types to `bind_default_call` and supply that result
as `call_binding` to the helper. A mapping from actual walks to `TorchProgram`
objects is specialized to local placement. A known missing walk is diagnosed
against the authored `when`; it never narrows the declaration silently.

`run(call)` returns a mapping of logical output names to tensors, or tuples of
tensors for per-request outputs. The helper checks fields and shapes, copies
into the generated call's newly allocated result storage, and returns the
model-owned result type. The logical output dtype stays with the binding.
Intermediate precision belongs to the implementation. A result is never an
unowned alias of a reusable workspace.

Each mode has its chosen number of workspace leases. Acquisition reserves
capacity without performing computation or communication. The engine closes
the lease at its completion boundary. Distributed agreement and resource
admission must finish before materialization or `run` can execute. The helper
does not own selection, collective ordering, resource rollback, or sampler
commit. Its initial supported device is CPU; this is not a generic asynchronous
Torch backend or a CUDA capture implementation.

## Logical weight preparation

`backends.weights.prepare_weights(binding, prefix, *, dtype=None, shards=None)`
loads the model's logical values and converts them into an immutable ordinary
tensor representation. With no explicit dtype it preserves the existing graph
helper's behavior. Identical representations share the binding's preparation
cache; aliases with identical partitions retain tensor identity.

An implementation can map a logical weight name to `TensorShard(axis, group="tp")`.
The helper obtains the role and group size from the binding and selects the
corresponding equal contiguous interval. It rejects a nondivisible extent.
The optimizer supplies the physical partition choice, not checkpoint names,
loading rules or rendezvous information. Replicated weights omit a shard entry.
Different partitions of one logical alias are legitimately distinct physical
representations. The optional ggml helper imports this same preparation function.

## The bounded Whisper CPU TP2 exercise

The deployment places the encoder on role 0 and the decoder on roles 0 and 1.
Use the generated FP32 logical interface. Prepare CPU weights in FP64 and use
FP64 arithmetic. Encoder results are copied to the generated FP32 output;
decoder cache writes target the runtime-owned FP32 aliases. Reads widen those
stored values for arithmetic. This explicitly chosen realization does not
qualify the older CPU FP32 graph interpreter or change numerical policy.

`implementations.whisper.torch.WhisperTorch` supplies optional encoder,
attention, normalization, embedding and decoder cache-access math. It consumes
the existing model configuration and admitted dense resource views. It performs
no request allocation, position advancement, sampling commit or process-group
initialization. Its default `feed_forward` is the unpartitioned encoder FFN.
The author supplies the decoder's `ffn(prefix, x)` callback.

For each decoder layer, partition `fc1.weight` and `fc1.bias` along axis 0 and
`fc2.weight` along axis 1. Preserve `fc2.bias` as a replicated value. Each role
computes GELU of its first projection, then its second projection **without**
the second bias. Combine contributions using the chosen collective and add the
second bias once afterward. Both roles perform necessary computation. The
helper supplies no finished distributed FFN or completed acceleration file.

The first decoder variant uses the bound TP group's `all_reduce`. The second
uses `all_gather` and sums the role dimension locally. Both advertise the same
bounded greedy domain, request replicated `DenseSequenceView` storage and use
explicit comparable selection costs. All participants must select one variant
and establish common workspace/resource admission before entering either
collective protocol. A refusal releases tentative peer reservations; an error
after execution entry is instance-fatal and cannot cause local substitution.

The installed-CLI generation, generated-to-filled diff, real two-process CPU
qualification and ordinary M* service campaign belong to the execution recipe.
Mechanism tests and algebra checks are separate evidence. Qualification remains
one centralized corpus consumer; this helper adds no model-specific test command.
