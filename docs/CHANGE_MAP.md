# Physical change map

The deliverable has two source repositories. The four returned Pi acceleration
files and three authored CPU TP2 books are present, along with the external
Qwen3-Omni Talker book. The delivery's receipts, manifests,
patches and Git bundles are evidence and transfer material, not a third runtime
project. The exact file list is also available in the generated patch inventory.

## 1. M* fork

### 1.1 Fresh construction and implementation binding

- `mstar/model/binding.py`: `NodeBinding` and derived `NodeAssembly`.
- `mstar/model/contract.py`: logical tensor/result descriptions consumed by the
  source generator and default call compiler. Runtime bindings carry preparation
  services and actual worker-local walks; dtype resolution is shared with generation.
- `mstar/model/base.py`: optional fresh `create_submodule` and logical-weight
  entry points and read-only `describe`; existing `get_submodule` remains the
  default deployment path. External implementations own TP/SP eligibility while
  existing signal-sharding boundaries remain intact.
- `mstar/model/submodule_base.py`: stable bound-execution environment, explicit
  device identity and typed-result adaptation.
- `mstar/worker/engine_manager.py`: opt-in fresh shell construction, static
  assembly, typed resource-request collection and ordinary resource binding. Bound
  programs do not first load/cast a reference compute stack.
- `mstar/engine/engine.py`: resource-ready program preparation and binding;
  consumes the implementation's execution request. A tensor-forward program
  remains a separate physical object and uses the existing Torch compilation,
  capture, slot and resource lifecycle according to its selected policy.

### 1.2 Execution capabilities and selection

- `mstar/engine/execution.py`: partial invocation facts, support status, options,
  acquired execution, actual stream/workspace context and typed results. Workspace
  acquisition receives invocation facts, permitting multiple modes in one option.
  Runner consumption uses the same support and admission gates; an execution
  kind does not become an additional semantic reference or scheduling predicate.
- `mstar/worker/batch_builder.py`: bounded proposed-size search using capability
  queries; legacy capture grouping remains available on the legacy path.
- `mstar/worker/micro_scheduler.py`: supplies the node's capabilities to the
  existing builder.
- `mstar/engine/engine.py`: logical-input refinement, one pre-effect execution
  reservation, resource admission, physical materialization and primitive call.
  Busy workspaces defer; unsupported invocations fail; result effects commit once.
- `mstar/engine/resources/step.py`: retryable `ExecutionBusy` admission outcome.
- `mstar/engine/accelerator_graph_runner.py`: a disabled-autocast scope actually
  disables ambient autocast; programs own their numerical policy.
  Native bound runners retain the ceiling replay bucket for their admitted row
  limit and omit larger unreachable buckets.

### 1.3 Logical weights and physical resource consumers

- `mstar/model/weights.py`: logical identities, aliases/defaults and source
  values before physical representation.
- `mstar/engine/resources/kv/config.py`, `resources/spec.py`: logical sequence
  geometry and deployment byte budgets retained until physical request resolution.
- `mstar/engine/resources/sequence.py`, `sequence_view.py`, `binding.py`: typed
  paged/dense and partition requests, logical use handles, compatible physical
  views and the shared serving/qualification resource constructor.
- `mstar/model/loader/base.py`, `loader/__init__.py`: separate existing logical
  checkpoint routing from target parameter loading; retain the legacy consumer.
- `mstar/engine/resources/base.py`: provider construction and returned-resource
  result binding; the resource remains the state owner.
- `mstar/engine/resources/convenience.py`: resource imports at actual use.
- `mstar/engine/resources/kv/manager.py`, `plan.py`, `cache.py`: injected physical
  storage with the existing page arena, admission/rollback/commit and owned plan
  snapshots. `kv/transfer.py` defers a type-only optional-backend import.
- `mstar/engine/resources/attn/planned.py`: attention-plan data for native
  consumers, including query/context/causality and page views.
- `mstar/engine/resources/position/manager.py`: host position-plan realization
  with the existing counters and fork semantics.
- `mstar/engine/resources/sampler/plan.py`, `resource.py`: sampling plan/result
  protocol within the current sampler lifecycle.
- `mstar/engine/resources/runner.py`: accept returned physical results before
  the existing commit.
- `mstar/engine/resources/kv/kernels.py`, `sampler/kernels.py`, `sampler/utils.py`:
  unchanged CUDA/Triton kernel bodies moved behind lazy imports so the physical
  provider can be selected without importing an unavailable GPU implementation.

### 1.4 Backend startup, completion and release

- `mstar/runtime_extensions.py`: trusted configuration factory and managed
  worker initialization/drain/close/fatal hold. No Pi model code in the core.
- `mstar/api_server/entrypoint.py`: carries an opaque admission context into the
  normal spawned conductor.
- `mstar/conductor/conductor.py`: explicit device selection; reserve native
  ownership before worker spawn; consume fatal status and release acknowledgement;
  retain uncertain workers instead of treating forced process death as a drain.
- `mstar/worker/worker.py`: backend health at worker boundaries, quiesce submitters
  and drain before ordinary failure cleanup, explicit orderly close. Existing
  Torch stream/event submission and CPU observation stay in place.
- `mstar/utils/orphan.py`: optional non-destructive stalled-orphan callback.
- `mstar/utils/device.py`: backend timing-event capability, separate from normal
  event completion. Pi advertises ordinary events, not event timing.
- `mstar/distributed/communication.py`: no communicator/backend lookup for a
  truly local TP1/SP1 worker.
- `mstar/communication/tensors.py`, `tensor_store.py`: close pending transport
  readers and release owned tensors after worker drain.
- `mstar/engine/engine.py`: teardown drops prepared programs, state/resources and
  the runner's ownership roots.

### 1.5 Whisper's model-owned boundary

- `mstar/model/whisper/whisper_model.py`: fresh encoder/decoder shells and logical
  weights beside the cached reference getter; config-only description and lazy
  preprocessing assets. Page budgets move into the deployment YAMLs.
- `mstar/model/whisper/submodules.py`: configuration-only construction, logical
  preparation, declared steps and result adaptation using the same graph outputs,
  text aliases, suppression and stopping rules. Encoder declares no stop buffers.
- `mstar/model/whisper/weights.py`: checkpoint names, aliases and zero defaults.
- `mstar/model/whisper/reference.py`: explicit original HF encoder and M* decoder
  construction for numerical qualification, plus retained reference capture/batch policy.
- `mstar/model/whisper/results.py`: typed encoder and decoder results.
- `mstar/model/whisper/config.py`: preserve distinct encoder/decoder architecture
  dimensions while sharing ordinary model configuration.
- `mstar/model/registry.py`, `mstar/api_server/openai/adapters.py`: `whisper` alias
  with a configured local checkpoint; ordinary OpenAI audio API adaptation.

### 1.6 Focused verification

- New modular tests: `test_bound_execution_integration.py`,
  `test_logical_weights.py`, `test_planned_resources.py`,
  `test_runtime_extension_lifecycle.py`.
- Existing test doubles in `test_engine_uniform_inputs.py` and
  `test_micro_scheduler.py` expose the new default `None` capability. Their
  existing expectations are unchanged.
- Existing scheduler, worker drain, admission, page/fork/prefix, sampling and
  Whisper sizing cases provide regression coverage. Host limitations reproduced
  on the unmodified baseline remain explicitly reported.

## 2. mstar_accelerators

### 2.1 Declaration and binding core

- `src/mstar_accelerators/definition.py`: the three-field declaration,
  implementation execution requests, preparation and prepared program shapes.
- `conditions.py`: small typed partially evaluable condition set and finite
  batch-bound extraction, canonical per-resource row conditions and bounded
  contradiction diagnostics. Diagnostics do not silently narrow authored support.
- `binding.py`: derive static and invocation facts from M*'s binding.
- `registry.py`: reference matching, candidate-specific local residual conditions, typed resource
  requests, preparation, capability construction, cleanup and hosted-walk progress
  checks. Irrelevant candidates cannot allocate or conflict over resources.
- `authoring.py`: additive CLI reading the actual model/deployment description
  and writing one editable source file per logical node. The author's explicit
  execution-template choice exposes preparation and runner decisions. Generated
  resource identities and logical Plans do not prescribe paged physical addresses.
- `sources.py`: file/directory loading with relative-path identities and optional
  retained module:symbol sources. Workers import files locally after spawn.
- `compiler.py`: generate the actual typed call from logical inputs, admitted
  logical uses and result descriptions; executable default callbacks with no verdict
  policy, optional helper requirement diagnostics and derived local preparation scope.
- `__init__.py`, `pyproject.toml`: public exports and package metadata.
- `tests/test_registry.py`: residual unknowns, supported-domain bounds,
  singleton coverage, preparation cleanup and selection ambiguity.

### 2.2 Pi hardware and Torch substrate

- `backends/pi/native/`: accepted Stage 2 canonical BO allocator, asynchronous
  submission, dependency/event snapshots, health and retirement. The accepted
  real-backing-extent repair is in the baseline, not reimplemented elsewhere.
- `backends/pi/torch/csrc/graph_provider.inc`, `ggml_provider.cpp`: complete actual
  ggml graphs, canonical intermediate arenas, preparation, slot acquisition,
  accepted-flight ownership, output/storage binding and checked retirement.
- `backends/pi/torch/mstar_pi/ggml.py`, `__init__.py`, `setup.py`: the real ggml
  Python graph/preparation surface and package registration/version.
- `backends/pi/third_party/ggml/`: managed complete-graph support and exact kernel
  variants required by the programs. The changed managed ABI, Vulkan provider
  and types, shader generator, argmax/copy/gather/norm/unary shaders are recorded
  against the accepted baseline. Normal ggml variants remain separate.
- `backends/pi/torch/mstar_pi/admission.py`, `tools/run_hardware.py`: managed
  process-tree ownership over the same established physical-device lease.
- `backends/pi/torch/tests/graph_host_preflight.py`, `check_graph_spirv.py`,
  `graph_consumer.py`: real metadata/shader checks and parent/spawn device gate.
- `backends/pi/tests/test_managed_admission.py`: descriptor/spawn/lease lifetime
  tests. Stage 1/2 fixtures and original gates remain unchanged.
- `backends/pi/kernel/`: retained status/retirement contract and original patch;
  this candidate requires the already verified r2 kernel, not a new kernel diff.
- Backend contracts, README and provenance are updated; old packet readers,
  manifests, obsolete handoffs and historical build receipts are delivery
  evidence rather than production source files.

### 2.3 M* adapter and shared physical binding

- `src/mstar_accelerators/backends/pi/runtime.py`: backend registration,
  initialization, registry assembly, health/drain/close, ownership and release
  receipts. It consumes M*'s optional runtime extension.
- `backends/graph.py`: optional implementation-selected graph preparation,
  weights, mode pools, resource-port binding, fresh results and explicit provider
  adaptation. Its keyword-only builder convention is typed; no provider is
  injected by runtime or qualification.
  Optional paged and dense input adapters share logical input/position staging;
  dense ports borrow the owner's contiguous aliases and append positions.
  Direct-cache graph computation remains optimizer-authored.
- `backends/paged.py`: explicit physical row maps. The obsolete external resource
  constructors and Pi compatibility aliases are removed; M* owns the common
  resource construction path.

### 2.4 Whisper authoring boundary

- `accelerations/`: the initial Package 1 exercise generated one encoder and one
  combined decoder, then deliberately specialized the decoder into paged prefill
  and contiguous decode sharing one persistent owner. The returned books are now
  present, with the later stochastic, CPU TP2 and AMD Talker implementations.
- `src/mstar_accelerators/implementations/whisper/pi_v3d/graph.py`: full encoder,
  prefill and decode using actual ggml primitives and canonical resource ports.
- The old handwritten Whisper `program.py`, `support.py` and three-entry registry
  are removed; their derivable binding work belongs to the shared compiler.
- `deployments/whisper_tiny_pi.yaml`: device, FP32, concurrency, resource sizes,
  placement and selected registry. Paths are rendered locally.

### 2.5 Qualification, environment and operation

- `catalog.py`: repository filesystem discovery, separate from deployment subsets.
- `testing/harness.py`: one registry-consumption loop, PASS/FAIL/SKIP reports,
  optional executable-reference adapters and an injected differential consumer.
- `testing/solve.py`: bounded, centralized witness search over declared
  alternatives and adapter-supplied realizable domains; unknowns remain diagnosed.
- `testing/numerics.py`: centralized, versioned component numerical policy,
  exact observations and retained historical pointwise diagnostic.
- `testing/models/whisper.py`: independent encoder/prefill/decode cases using
  model-owned CPU references, actual registered preparation and M* state plans.
- `testing/pi.py`, `testing/__main__.py`: configurable qualification driver and
  CLI, optional prerequisites, existing managed-worker drain/release ownership.
  No test verdict enters M* serving or the declaration's three-field identity.
- `src/mstar_accelerators/testing/whisper.py`: shared input, observation,
  resource and lifetime helpers. The former standalone model qualification
  entry point is removed; acceleration verdicts come from corpus consumption.
- `testing/reference_resources/`: explicit CPU reference resources and plan
  snapshot consumer; not runtime fallback implementations.
- `tests/whisper/`: exact graph evaluator; original reference comparisons;
  one-/two-request real engine integration; full-checkpoint numerical gate;
  test-only CPU runtime for a real server rehearsal.
- `tests/fixtures/whisper/`: fixed audio, model asset identities and frozen
  token/state/text oracles with provenance.
- `tools/whisper/prepare_assets.py`: pinned download/verification and deployment
  rendering; `demo.py`: actual API/conductor/worker/client gate with streaming,
  concurrent requests, cancellation, rejection, recovery and acknowledged close.
- `requirements/whisper-pi.txt`, `tools/environment/`: tested dependency choices
  and reproducible retained TorchCodec source setup for the AArch64 environment.
- Root/backend guidance, architecture and this map keep the responsibilities and
  current supported domain reviewable. Generated runs stay outside both repos.

The source-generation rehearsal changes logical description, default compilation,
file discovery and acceleration binding. Its native Pi runtime, Torch provider,
ggml kernels and central numerical policy remain unchanged. Host tests include
actual generated files, filesystem specialization, complete real-engine EOS
trajectories and the ordinary seven-case M* service with an explicit CPU graph
interpreter. Prior Pi success remains separately identified evidence; new bindings
require a fresh target run. Stored-capture replay never implies fresh execution.


## 3. Package 2 over the fresh Package 1 target return

M* remains at `c9a1e20cfc5a30597e617f801d053668f6e56aeb`. The new external
foundation starts from returned `9f535df3af0d2c705b9f23efe3844c53617f7e34`.
The preceding map describes the historical foundation changes; the three
Package 1 authored corpus files are now present and retained byte-for-byte.

- `testing/models/whisper.py`: separate exact survivor self/cross/position/sampler
  preservation from independent numerical accuracy; retain the earlier actual
  pre-decode cross-state snapshot repair. A separate commit and RED/GREEN evidence
  identify the survivor repair.
- `backends/pi/third_party/ggml`, `torch/csrc/graph_provider.inc`: optional native
  categorical graph operation using stateless Philox/Gumbel arithmetic, F32
  logits/temperatures and I64 seed/offset/result ports. Existing managed stream,
  arena, output and retirement mechanics remain the owners. The userspace native
  package becomes `mstar-pi` 0.4.0; rebuild it. Kernel/ICD/admission are unchanged.
- `backends/pi/docs/categorical.md`, `torch/tests/categorical_*`: operation policy,
  original RNG vectors and focused primitive parent/spawn conformance. Existing
  backend gate cases retain their meanings.
- `backends/graph.py`: optional sampler-plan staging shared by both physical
  cache adapters; `categorical_tokens` supplies the optional physical graph port
  convention. No sampler lifecycle or hidden input-policy restriction is added.
- `implementations/whisper/pi_v3d/{graph,dense}.py`: optional sampling callback
  after unchanged model suppression and decoder math. Greedy defaults remain.
- `testing/sampling.py`, `testing/numerics.py`, `testing/models/whisper.py`:
  central canonical-policy witnesses, bounded independent distribution checks,
  exact seeded replay and real owner effects. A combined declaration may supply
  its own prefill in the bounded lifecycle trajectory. No alternate consumer.
- `tests/whisper/evaluator.py`: categorical support only in the explicit host
  interpreter; source imports a test-only scalar reference, never runtime fallback.
- `tools/whisper/demo.py`: retained seven greedy cases plus optional seven
  positive/mixed-policy replay, streaming and cancellation cases. Unsupported
  sampling now requests positive temperature with top-p filtering, since ordinary
  positive temperature is the new supported workload.

Package 2 supplies these mechanisms but leaves the additional stochastic book
for authentic installed-CLI generation and authoring. New native compilation,
central whole-corpus qualification, ordinary mixed serving, traced qualification
and final environment/owner receipts belong to its target execution recipe.
This section describes the historical Package 2 supplied foundation. Its returned
final source is M* `156e640c` / external `2671a49e`, including the authored
stochastic decoder and separately committed repairs. Package 3 starts there.

## 4. Package 3 ordinary Torch and distributed execution

- `backends/weights.py`: shared logical-weight conversion and typed role-local
  tensor partitioning, with alias preservation and immutable representation reuse.
  `backends/graph.py` imports the same function; its existing programs are retained.
- `backends/torch/programs.py`: optional synchronous CPU callback/slot helper
  consuming generated calls and copying logical outputs into owned result storage.
- `backends/torch/runtime.py`: production CPU registry adapter; no Pi admission,
  device-predicate projection, graph interpreter or reference compute stack.
- `registry.py`: derive candidate execution names for M*'s pre-allocation
  agreement without another optimizer-maintained identity field.
- `implementations/whisper/torch.py`: optional FP64 encoder/attention/decoder
  math over FP32 logical inputs/outputs and runtime-owned contiguous cache aliases.
  FFN partition/collective computation remains the optimizer's callback.
- `deployments/whisper_tiny_cpu_tp2.yaml`: rank-0 encoder and TP2 decoder,
  two CPU workers, existing resource budgets and explicit ordinary Torch runtime.
- `docs/TORCH_IMPLEMENTATIONS.md`: program, weight partition and authored
  collective boundaries. The CPU corpus now contains the generated-to-filled
  encoder, all-reduce decoder and gather/local-sum decoder.
- `tests/test_torch_programs.py`, `tests/whisper/test_torch_math.py`: focused
  weight/alias/FFN algebra, slot/result lifetime and independent small-model
  numerical/state checks. These are host mechanism evidence, not target TP2
  qualification or a replacement acceleration consumer.

The M* distributed consumer owns cross-rank selection, workspace/resource
admission rollback and instance-fatal propagation. The centralized qualification
environment supplies actual roles; ordinary service supplies separate integration
evidence. The original Package 3 foundation left the books for its target recipe;
the returned source includes those authored files and preserves the exact
installed-CLI starting point under `authoring/package3_cli_generated`.

## 5. AMD A with explicit execution requests

This revision preserves the AMD A experiment's physical starting-point goal
while separating the external physical program from M*'s logical model shell.
It replaces the earlier shell-installing adapter; the logical reference no longer
serves as a physical-program factory for the acceleration.

- `mstar/engine/tensor_forward.py`: runtime-owned compile/capture policies,
  prepared program carrier and runner, consumed through ordinary execution
  options. Physical output collection precedes logical result adaptation.
- `mstar/engine/engine.py`, `accelerator_graph_runner.py`: selected physical
  program consumption, shared warmup/resource ordering, capture/replay, stable
  output ownership and actual execution-path receipts.
- `mstar/model/binding.py`, `engine/resources/sequence.py` and resource binding:
  explicit device-plan requests are visible before resource construction. Runner
  choice itself does not imply a storage representation or plan location.
- `mstar/model/qwen3_omni/logical.py`: importable Talker logical identity, request
  interpretation, resource declarations and result semantics without constructing
  its physical program or importing the device-kernel stack.
- `definition.py`, `registry.py`: `Implementation.execution` accepts an explicit
  `TensorForwardExecution`; preparation returns an independent physical module.
  The registry binds the request to the runtime runner and preserves the existing
  `when`, resource-request, selection and cost consumers.
- `compiler.py`: contract-derived tensor-result packing and reconstruction.
  Physical programs name logical result fields; the model maps those results onto
  graph edges and effects.
- `backends/torch/tensor_forward.py`: optional typed physical resource choices
  for the runtime's numerical methods and device plans. These do not create a
  second resource allocator or state owner.
- `implementations/qwen3_omni/amd/`: external Talker numerical program, weights,
  physical packing and capture recipes, with source provenance. The filled Talker
  book selects that program and its explicit compile/capture policy.
- `authoring.py`, `tests/test_authoring.py`: real CLI selection of callable or
  tensor-forward templates, preserving one file per node, logical typed records,
  correlated support and executable placeholders. Device choice does not select
  a template. Exact installed-CLI output and generated-to-filled diffs are retained
  separately from the active corpus.
- `docs/EXECUTION_IMPLEMENTATIONS.md`: the execution producer/consumer contract
  and its ownership boundaries. Current-corpus guidance now identifies the filled
  CPU books instead of describing their historical empty destination.

The AMD deployment still selects only Talker externally. Host generation and
runtime-mechanism tests are evidence for their stated boundaries; actual ROCm
serving, captured Qwen execution and any benchmark require target hardware.
