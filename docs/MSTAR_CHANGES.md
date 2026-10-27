# Changes in the M* fork

This maps the integrated fork pinned by [MSTAR_REVISION](../MSTAR_REVISION).
Changes are grouped by responsibility rather than hardware or development history.
The fork is based on upstream `887c499fc0c7bc3eae249743c01208b7eb3a6a3e`;
[review the complete diff](https://github.com/rohanchanani/mstar/compare/887c499fc0c7bc3eae249743c01208b7eb3a6a3e...555f4845cbe878890e2fb2d5eed01dd6528a6892).

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
scheduler, KV allocator, sampler or result-lifetime policy. Physical resource requests resolve through the existing resource owners.

## Optimizations of existing runtime machinery

- **Preparation and scheduling:** retain interpreted rows through admission
  retries; preserve the overlap between resource commit and output collection.
  Fixed implementation selection is an optimization justified by a positive
  support proof, not a new scheduling policy.
- **Host resource planning (Pi):** materialize host KV tensors only when a consumer
  needs them, retaining immutable plan snapshots. This coexists with AMD's
  explicit device-plan requirements without forcing device synchronization.
- **Result delivery:** retain upstream's per-request collector, disconnect
  handling, transport and error policy. Managed admission and drain notifications
  use those mechanisms; Whisper's inline signals remain model-owned.

Prepared predicate evaluation, precomputed field/shape binding and CPU destination
identity shortcuts live in **mstar_accelerators**, not in the M* fork. The shared binder retains
these optimizations alongside AMD's dynamic row contracts.

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
  ownership and permit the established recovery path. Managed teardown uses upstream transport shutdown and acknowledges release
  only after successful drain and close.

## Model and API integration

M* retains logical Qwen and Whisper definitions: graph/walk topology, checkpoint
identities, request interpretation, resource effects and result mapping. The
external implementations own physical weights, packing, kernels and programs.

Keep the portable media decoding needed by AMD and the Pi PCM/Whisper frontend
changes with their existing API and model owners. Hardware support must not
silently select a different model interpretation, execution API or fallback.

## Verification

The centralized harness checks both reference and acceleration through M*.
Deployment fidelity is consumed centrally; Whisper's declared quality baseline
does not alter Qwen's default contract. Binding, admission, resource and lifetime
checks cover their respective runtime boundaries. Service checks and performance
measurements are separate: component timings do not replace the final benchmark.
