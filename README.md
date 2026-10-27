# mstar_accelerators

Handwritten implementations of M* graph nodes, declared as
`Acceleration(reference, when, implementation)`.
The model defines the computation and state effects; the optimizer supplies the
physical implementation; M* owns serving, resource lifetimes and state commits.

**Authoring an acceleration?** Start with the examples below. You do not need to
understand the device backend first. **Running a model?** See [How do I run it?](#how-do-i-run-it).

## Available accelerations and results

| Model | Hardware | Input → output |
| --- | --- | --- |
| Whisper-tiny (multilingual) | Raspberry Pi 5 | Audio → text |
| Qwen3-Omni-30B-A3B | AMD Instinct MI300X | Text, image, audio → text and speech |

### Whisper-tiny on Raspberry Pi 5

| Warm HTTP transcription latency | ggml whisper.cpp + Arm KleidiAI | Accelerated M* |
| --- | ---: | ---: |
| Mean of per-clip medians | 460.3 ms | **325.5 ms (−29.3%)** |

Same Pi, full encoder context and greedy decoding. Six fixed
English development clips, four repetitions per implementation, models already
loaded. The public baseline uses Q8_0, four threads and Flash Attention. Common
normalized WER was 11.5% for the baseline and 8.0% for M*.

### Qwen3-Omni on one MI300X

Cell: **time-to-first-audio change / audio-throughput change**, versus unchanged
`users/atindra/amd-native` (`678f59d6`). Negative latency and positive throughput
are better; `c` is client concurrency.

| Input → speech | c1 | c4 | c16 | c32 |
| --- | --- | --- | --- | --- |
| Text | -8.4% / +16.7% | -12.8% / +15.0% | -13.3% / +15.2% | -12.9% / +15.5% |
| Image | -14.2% / +19.1% | -17.1% / +18.4% | -13.0% / +17.5% | -15.7% / +17.7% |
| Audio | -17.0% / +20.8% | -15.0% / +15.3% | -13.5% / +12.9% | -18.0% / +17.1% |

## What does an optimizer write?

Start with the Pi CPU encoder, `accelerations/whisper/pi_cpu/audio_encoder.py`:
typed call records, a preparation function and one declaration. Then read the
combined decoder, `accelerations/whisper/pi_cpu/decoder.py`, which adds
prefill/decode support, sampling conditions and runtime-owned KV views:

```python
acceleration = Acceleration(
    reference=WhisperDecoderSubmodule,
    when=When(common, sampling_policy, CPUTokenDecoderAvailable(readout_format="q4_0_kai")),
    implementation=Implementation(
        prepare=partial(prepare, sample_tokens=categorical_token, readout_format="q4_0_kai"),
        resources=resource_requests,
    ),
)
```

The [declaration types](src/mstar_accelerators/definition.py) are:

```python
@dataclass(frozen=True)
class Acceleration:
    reference: type
    when: Condition
    implementation: Implementation
```

`reference` is a class treated as a **logical NodeSubmodule**: it identifies the
model-owned computation and state effects, independently of its physical implementation.
An acceleration must satisfy its semantic contract wherever `when` holds:

$$
\llbracket n\rrbracket_{W,C}:
I_n \times S_n \to O_n \times S_n \times E_n
$$

Inputs and state produce outputs, updated state, and effects.

`when` describes supported facts through the [Condition protocol](src/mstar_accelerators/conditions.py):

```python
class Condition(Protocol):
    def evaluate(self, facts: Mapping[str, Any]) -> Support: ...
    def specialize(self, facts: Mapping[str, Any]) -> "Condition": ...
```

These facts may be consumed statically during assembly into M* or dynamically by
the M* runtime (scheduler, engine, resource manager). `Support` is YES, NO or
UNKNOWN; missing facts remain unknown. `When(...)` combines conditions with AND.

`implementation` supplies the **physical realization** of that logical node:
executable computation, its preparation, and required views of runtime-owned
resources. It may specify an execution adapter and a selection cost. M* owns
scheduling, resource lifetimes and state commits.

To author another implementation, use the installed CLI:

```sh
mstar-accelerators create --deployment /absolute/model.yaml \
  --out-dir /absolute/new-implementation
```

It derives logical call records from the model without loading weights or
overwriting files. Fill in support, physical views and computation. The default
interface supplies `primitive(call, execution)`; `--execution tensor-forward`
selects the tensor-program template. Device identity does not select the interface.

## How do I test an acceleration?

Run the centralized testing harness against a registry source, for example:

```sh
python -m mstar_accelerators.testing \
  --registry accelerations/qwen3_omni/amd \
  --report /absolute/report.json --require-all-pass
```

`--registry` accepts a declaration file, a directory or a `module:symbol` mapping
of selected entries, so the harness can consume arbitrary subsets of the
acceleration registry. There is no test configuration or driver selection.
`--require-all-pass` requires complete PASS coverage; unavailable coverage is
reported as SKIP, not success.

Conceptually, the harness follows this procedure:

```text
environments = discover_hardware_and_runtime_capabilities()
for acceleration in selected_registry:
    for node, deployment in corpus.match(acceleration.reference):
        fidelity = required_fidelity(deployment)
        cases = construct_cases(node, environments, acceleration.when, coverage_policy)
        for case in cases:
            reference = M.execute_reference(node, case)
            candidate = M.execute_acceleration(acceleration, case)
            report.record(verify(node.semantics, fidelity, reference, candidate))
report.include_generated_plan_and_coverage()
```

Both executions start from the same logical inputs and state, with independent
resources owned by M*. `verify` is responsible for enforcing the acceleration's
compliance with the node's semantic contract:

$$
\llbracket n\rrbracket_{W,C}:
I_n \times S_n \to O_n \times S_n \times E_n
$$

Inputs and state produce outputs, updated state, and effects. Enforcing this
contract is a broad, difficult engineering problem. Our first pass combines
PyTorch's numerical comparator with reference-based generation checks and
end-to-end speech tests, while checking that execution preserves the required
state and resource behavior. Together, these check numerical agreement and
generated-output quality on the tested cases; they do not prove equivalence
for every possible input.
Verification itself is a shared, centralized engineering artifact: a growing
knowledge base for enforcing this variable semantic contract, rather than an
independently generated test for each new workload. Workloads may expose correct
programs that fail verification or buggy programs that pass; both should drive
justified PRs to the repo-wide verifier. The goal is a maintainable tool that
converges toward an increasingly better approximation of a universal semantic
equivalence checker. See [qualification](docs/QUALIFICATION.md) for the current
procedures and coverage limits.

Deployment authors optionally specify required fidelity as some notion of the
minimum quality/performance an acceleration must satisfy. The centralized harness
owns how that requirement is verified.

```yaml
fidelity:
  mode: quality
  at_least: public-q8-kleidiai-v1
```

The model declares the baseline's pinned checkpoint, inputs, decoding protocol
and measured quality. The harness consumes it, executes the acceleration with
reference continuations through M*, and applies the centrally implemented metric.
This does not relax state, lifetime, shape, finite-value or sampling obligations.
`autocast_dtype` retains its execution/representation meaning; it is not an
accuracy tolerance. No fidelity fields or test callbacks belong in an acceleration.
The generated report records the deployment requirement and coverage.
The current fidelity interface supports numerical and baseline-quality requirements;
performance requirements remain a future extension.

Verification is a shared engineering artifact of the company: individual
optimizers do not own their own testing or acceptance criteria. New accelerations
consume the existing harness; new model semantics, hardware and coverage extend
its shared machinery. Changes to the harness—such as accommodating differences
justified by numerical precision or fixing checks that accept incorrect
accelerations—must be separate, justified PRs, independently reviewed rather than
tailored to make one implementation pass.

## What happens to that declaration?

Both examples cross the same ownership boundaries:

| Step | Code to read | Responsibility |
| --- | --- | --- |
| Select source | [TorchRuntime.assemble](src/mstar_accelerators/backends/torch/runtime.py), [AMD deployment](deployments/qwen3_omni_amd.yaml) | Deployment selects files and placement; the worker loads them. |
| Assemble candidates | [Registry.assemble](src/mstar_accelerators/registry.py) | Specialize support and collect physical requests without allocating logical state. |
| Resolve resources | [Resource ownership](docs/ARCHITECTURE.md#3-support-scheduling-and-resource-ownership) | M* resolves compatible representations and allocates shared resources. |
| Prepare and bind | [Compiler](src/mstar_accelerators/compiler.py), implementation preparation | Prepare weights/native code and bind model-derived inputs, outputs and admitted plans. |
| Acquire and execute | [Callable helper](src/mstar_accelerators/backends/torch/programs.py), [tensor-forward interface](docs/EXECUTION_IMPLEMENTATIONS.md) | M* admits resource effects and executes using the selected context and workspace. |
| Publish, commit, release | [Execution ownership](docs/EXECUTION_IMPLEMENTATIONS.md) | Return durable, ready outputs; M* commits model-declared effects and releases settled work. |

The Pi CPU decoder shares a native body and workspace capacity across prefill and
decode. V3D submits asynchronously through the Pi backend; AMD uses M*'s existing
Torch graph runner. None introduces another request scheduler, KV allocator or
sampling lifecycle.

## How do I run it?

Use editable installations of this repository and the M* revision recorded in
[MSTAR_REVISION](MSTAR_REVISION). The corpus, deployment templates and tools live
in the checkout, not the Python wheel.

- **Pi 5:** [Pi walkthrough](docs/PI5.md) covers the CPU build, assets,
  qualification and ordinary serving. [V3D backend documentation](backends/pi/README.md)
  covers its native execution contract and managed launcher.
- **AMD MI300X:** [Qwen3-Omni walkthrough](docs/QWEN3_OMNI_AMD.md) covers pinned
  ROCm dependencies, checkpoint preparation and ordinary M* serving.

[Central qualification](docs/QUALIFICATION.md) consumes the same declarations
and checks numerical results, state effects and lifetimes. Missing assets or
hardware produce explicit SKIPs. Serving never depends on a qualification verdict.

## Changes to M*

The integrated fork changes fall into four categories. The source map below
describes their owners in the paired runtime.

| Category | What belongs in M* | Main consumers / source |
| --- | --- | --- |
| Acceleration integration | Logical model/weight descriptions; candidate support and workspace admission; physical resource requests; execution and distributed agreement. | [`NodeBinding` / `NodeAssembly`][binding], [`ExecutionCapabilities`][execution], [engine][engine], [resource binding][resources], [`ExecutionAgreement`][agreement]. |
| Runtime optimizations | Retain prepared inputs across retries; eliminate selection only when a fixed implementation is proved; preserve commit/collection overlap. Host KV plans are materialized lazily; result delivery retains upstream’s collector and disconnect handling. | [Engine][engine], [KV planning][kv], `api_server/data_worker.py`, `conductor/conductor.py`, `worker/worker.py`. |
| Resource correctness and hardware support | Preserve positions and RNG offsets across repeated registration and eager/captured execution; provide ROCm attention, norm, RoPE and sampling implementations. Pi adds explicit KV storage encoding and backend lifetime integration. | [Sampler][sampler], [positions][positions], [resource views][views], [platform dispatch][platforms]. |
| Model and API integration | Logical Qwen/Whisper shells, checkpoint identities, result mapping, media decoding and transcription/speech endpoints. | [Logical Qwen model][qwen], [logical contracts][contracts], `model/whisper/`, `api_server/`. |

Model-specific optimized programs and kernels stay in **this** repository.
Generic resource algorithms and serving machinery stay in **M***. The registry,
generator, binder and qualification harness also live here; they are not another
server or another source of request state. [Detailed runtime responsibilities](docs/MSTAR_CHANGES.md).

[binding]: https://github.com/rohanchanani/mstar/blob/555f4845cbe878890e2fb2d5eed01dd6528a6892/mstar/model/binding.py
[execution]: https://github.com/rohanchanani/mstar/blob/555f4845cbe878890e2fb2d5eed01dd6528a6892/mstar/engine/execution.py
[engine]: https://github.com/rohanchanani/mstar/blob/555f4845cbe878890e2fb2d5eed01dd6528a6892/mstar/engine/engine.py
[resources]: https://github.com/rohanchanani/mstar/blob/555f4845cbe878890e2fb2d5eed01dd6528a6892/mstar/engine/resources/binding.py
[agreement]: https://github.com/rohanchanani/mstar/blob/555f4845cbe878890e2fb2d5eed01dd6528a6892/mstar/distributed/execution.py
[kv]: https://github.com/rohanchanani/mstar/blob/555f4845cbe878890e2fb2d5eed01dd6528a6892/mstar/engine/resources/kv/plan.py
[sampler]: https://github.com/rohanchanani/mstar/blob/555f4845cbe878890e2fb2d5eed01dd6528a6892/mstar/engine/resources/sampler/resource.py
[positions]: https://github.com/rohanchanani/mstar/blob/555f4845cbe878890e2fb2d5eed01dd6528a6892/mstar/engine/resources/position/manager.py
[views]: https://github.com/rohanchanani/mstar/blob/555f4845cbe878890e2fb2d5eed01dd6528a6892/mstar/engine/resources/sequence.py
[platforms]: https://github.com/rohanchanani/mstar/blob/555f4845cbe878890e2fb2d5eed01dd6528a6892/mstar/platforms/interface.py
[qwen]: https://github.com/rohanchanani/mstar/blob/555f4845cbe878890e2fb2d5eed01dd6528a6892/mstar/model/qwen3_omni/thinker_logical.py
[contracts]: https://github.com/rohanchanani/mstar/blob/555f4845cbe878890e2fb2d5eed01dd6528a6892/mstar/model/contract.py
