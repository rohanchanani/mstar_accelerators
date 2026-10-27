# Changes in the M* fork

This is a source map for integrating the AMD and Pi branches. It groups changes
by responsibility, rather than by hardware or the order in which they were made.
It is not a claim that the branches have already been merged.

## Integration contracts and consumers

| Producer / declaration | Consumer in M* | Responsibility |
| --- | --- | --- |
| Model descriptions and logical weights | `model/base.py`, `model/contract.py`, model logical shells | Define inputs, outputs, weights and state effects independently of the selected physical program. |
| `Registry.assemble(NodeBinding)` | `worker/engine_manager.py`, `model/binding.py` | Construct fresh logical shells, gather physical requests, prepare programs and transfer ownership. |
| Derived `ExecutionCapabilities` | `engine/execution.py`, `engine/engine.py`, `worker/worker.py` | Select supported invocations, reserve workspace, distinguish unsupported work from temporary backpressure. |
| `Implementation.resources` | `engine/resources/binding.py`, `sequence.py`, resource owners | Resolve compatible storage, encodings and views; retain allocation, budgets, admission and commit in M*. |
| Optional `TensorForwardExecution` | `engine/tensor_forward.py`, `accelerator_graph_runner.py` | Consume physical packing and capture recipes through the existing runtime graph/slot lifecycle. |
| Candidate identities and admission outcomes | `distributed/execution.py` | Agree across actual participants before effects or collectives. |
| Prepared/native work ownership | `runtime_extensions.py`, worker/engine shutdown | Drain work and retain ownership until release succeeds. |

Callable and tensor-forward programs adapt into the same execution contract.
Their different physical requirements justify an explicit adapter, not a second
scheduler, KV allocator, sampler or result-lifetime policy. Legacy resource-provider
overrides should not survive alongside typed requests for the same responsibility.

## Optimizations of existing runtime machinery

- **Preparation and scheduling:** retain interpreted rows through admission
  retries; preserve the overlap between resource commit and output collection.
  Fixed implementation selection is an optimization justified by a positive
  support proof, not a new scheduling policy.
- **Host resource planning (Pi):** materialize host KV tensors only when a consumer
  needs them, retaining immutable plan snapshots. This must coexist with AMD's
  explicit device-plan requirements without forcing device synchronization.
- **Result delivery (Pi):** preserve Whisper's inline client signals and the API,
  conductor and worker notification/wakeup improvements. Model-specific opt-ins
  must not change Qwen's routing or completion semantics.

Prepared predicate evaluation, precomputed field/shape binding and CPU destination
identity shortcuts live in **mstar_accelerators**, not in the M* fork. Merge their
optimizations into the shared binder while retaining AMD's dynamic row contracts.

## Correctness and hardware support

- **State continuity:** sampler and position ingestion must preserve a resident
  request's counters. Eager and captured execution must use the same authoritative
  RNG-offset state and advance it exactly once through the owning resource.
- **Physical representations:** Pi's KV encoding and AMD's device planning are
  separate properties of one resource contract. Encoding-aware budgets,
  fingerprints and conflict checks remain with the resource owner.
- **Generic ROCm implementations:** attention, RoPE, normalization, sampling,
  compilation and platform dispatch remain reusable M* mechanisms. Qwen-specific
  fusions and numerical programs stay in the acceleration repository.
- **Lifetime:** successful close is the release boundary. Failed close must retain
  ownership and permit the established recovery path. The audited AMD
  mark-closed-before-callback guards need reconciliation before integration;
  copying them unchanged would weaken Pi's failure semantics.

## Model and API integration

M* retains logical Qwen and Whisper definitions: graph/walk topology, checkpoint
identities, request interpretation, resource effects and result mapping. The
external implementations own physical weights, packing, kernels and programs.

Keep the portable media decoding needed by AMD and the Pi PCM/Whisper frontend
changes with their existing API and model owners. Hardware support must not
silently select a different model interpretation, execution API or fallback.

## Merge verification

Preserve both measured source pairs. Combine shared contracts and consumers
explicitly, then run existing binding, admission, resource, lifecycle and authoring
checks. Central qualification remains one harness with explicit model-specific
numerical policies: Whisper's quantization allowances are not Qwen defaults.
Verify each target through ordinary M* serving before benchmarking the final
combined artifact. Component timings do not replace that end-to-end comparison.
