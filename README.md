# mstar_accelerators

Implementation declarations and binding for M* nodes. The same acceleration
definition is the unit an optimizer develops and a deployment ships:

```python
Acceleration(reference=WhisperDecoderSubmodule, when=..., implementation=...)
```

The model owns logical inputs, weights, architecture, graph walks and state
transitions. An implementation supplies its supported domain, physical
preparation, program and execution request. M* binds the declaration to a fresh
logical node, selects a supported execution, admits its resources, runs the
physical program on the worker's Torch stream and commits the model's declared
effects.

## Run AMD A from the paired source package

The package contains sibling `mstar/` and `mstar_accelerators/` source trees.
From the extracted package root on the configured gfx942 AMD host:

```bash
cd mstar_accelerators
bash tools/qwen3_omni/run_a.sh ../mstar /absolute/path/to/amd-work /opt/rocm-7.2.0
```

This installs the pinned environment, runs target preflight, verifies and renders
checkpoint assets, requires complete central Talker qualification, and records
the full service demo. Fresh logs and receipts go under `amd-work/runs/`.
[QWEN3_OMNI_AMD.md](docs/QWEN3_OMNI_AMD.md) is the complete runbook, including
host requirements, an existing-checkpoint option, individual commands, evidence
scope and the authoring interface. The same command accepts an existing paired
Git checkout or the delivered source archive.

## Current corpus and authoring

The returned Package 2 corpus contains the encoder, greedy paged prefill,
greedy contiguous decode and combined native stochastic decoder. Those four
declarations are retained unchanged. They share the model's logical references
and runtime-owned persistent resource and sampling lifecycle.

The authored CPU corpus contains an encoder and two combined decoder variants.
They use ordinary Torch callbacks, keep attention/KV replicated, partition FFN
intermediate channels, and choose all-reduce or gather/local-sum collectives.
Their original installed-CLI snapshots remain under `authoring/`. See
`docs/TORCH_IMPLEMENTATIONS.md` for the physical helper and ownership boundary.

The Qwen3-Omni AMD A deployment selects an external Talker implementation. A is
the physical starting point for the experiment: its acceleration owns the Talker
program and asks M* to execute its tensor forward program with the selected
compile/capture policy. The remaining four nodes use the model's existing
deployment path. See `docs/EXECUTION_IMPLEMENTATIONS.md` for the runner interface
and `docs/QWEN3_OMNI_AMD.md` for target bring-up. Source and host checks alone are
not an AMD serving or performance result.

The first deployment target is FP32 Whisper Tiny on one Pi 5 V3D device: a complete
encoder, four-token prefill and one-token cached decode. Executions have one row;
the deployment admits two concurrent requests. M* retains preprocessing,
graph traversal, page allocation, positions, sampling bookkeeping, stopping,
streaming, cancellation and transport. The native program implements the actual
tensor computation. No reference compute stack is built in the Pi worker.

The fresh Package 2 return at M* `156e640c` and external `2671a49e` reports four
native corpus PASS registrations / 1,268 comparisons and all fourteen ordinary
service cases passing after separately committed repairs. Its original failures,
recoveries and exact source remain predecessor evidence. Each changed physical
implementation needs evidence from its own target campaign. See
`docs/QUALIFICATION.md` for the central consumer; no speed claim is made here.

## Repository map

- `src/mstar_accelerators/`: the declaration, condition and binding library.
  `implementations/whisper/pi_v3d/graph.py` contains the compute builders.
  `compiler.py` and `backends/` supply shared call/resource/program binding.
  `testing/` contains explicit reference and numerical qualification consumers.
- `accelerations/`: ordinary editable source files, one exported `acceleration`
  per file. The Pi and CPU Whisper books and AMD Talker book share this format.
  The CLI creates one file per logical node, covering its hosted walks.
- `src/mstar_accelerators/catalog.py`: discovery of that repository corpus,
  separate from deployment source subsets. No registration index needs editing.
- `backends/pi/`: the native runtime, real Torch backend, pinned ggml source,
  kernel contract, provider and backend gates. It does not import M* or own
  model requests.
- `deployments/`: reviewable deployment configuration, with generated local
  checkpoint paths supplied by the asset tool.
- `tools/`: environment setup and actual M* demo orchestration.
- `tests/`: interface tests, model tests and fixed audio/oracle identities.
- `docs/`: ownership, scope and the physical change map.

Read `docs/ARCHITECTURE.md` for the interfaces and `docs/CHANGE_MAP.md` for the
two-repository change hierarchy. Delivery-specific recipes pin the source pair,
environment, build and evidence gates; historical recipes describe their own
source versions.

## Generate, fill, and select acceleration files

From this repository, after installing it editable and the matching M* fork
(the default corpus location belongs to this checkout):

```bash
mstar-accelerators create --deployment /absolute/path/to/whisper.yaml \
  --out-dir /absolute/path/to/new-whisper --execution callable
```

The existing deployment YAML already names its model and configuration. The
command reads that description without loading weights or preprocessing assets,
then writes `audio_encoder.py` and `decoder.py`. It never edits its inputs or
overwrites an existing output. The generated files contain typed inputs, logical
resource-use handles, resource identities and calls, an executable placeholder,
and the derived initial `when`.
No runtime field labels a file unfinished; the testing consumer owns its verdict.

The author chooses the execution template. `callable` is the default regardless
of the deployment device. For an implementation using the tensor forward runner:

```bash
mstar-accelerators create --deployment /absolute/path/to/qwen3-omni.yaml \
  --nodes Talker --out-dir /absolute/path/to/new-talker --execution tensor-forward
```

This template emits an independent physical module with input packing, forward
and capture-recipe hooks. Its `TensorForwardExecution` request exposes compile,
capture and autocast choices; compile and capture start disabled. The resource
request helper is explicit and editable. The logical shell remains unchanged.
See `docs/EXECUTION_IMPLEMENTATIONS.md` for the exact prepared-program and output
conversion boundaries. `--nodes` overrides the deployment's node selection.

Fill the selected physical program and state its actual support. The default
materializer derives the call from model contracts and admitted runtime plans.
`Implementation.resources` selects typed physical views of the node's logical
resources. Its generated paged choice is editable; neither the logical model nor
the generated `Plans` commits to a paged address ABI. The resource owner resolves
compatible requests together before programs prepare.

Preserve the exact generated files outside corpus discovery and keep the
generated-to-filled diff before adding a filled book under `accelerations/`.

Arbitrary callbacks remain supported. The existing optional ggml helper is
ordinary implementation code: an acceleration choosing it supplies its graph
builder and provider explicitly. The CLI, registry and Pi runtime do not choose
that programming model or inject a graph provider.

To create another variant, copy the file and edit its `when` and implementation.
Its relative path supplies registration identity. A deployment selects a file or
directory in `runtime_extension.accelerations`; paths are resolved from the
launch working directory. The asset tool writes an absolute source path. Files
load in the consuming process, so generated Python callbacks are not pickled
from the conductor into a spawned worker. Shared compute code uses ordinary
absolute imports.

The first decoder file covers both prefill and decode. Multiple mode programs
inside it share one submodule, logical resources and prepared weights. Copying a
file adds a selectable implementation to that same node, not a new model node.

## Centralized qualification

`run_registry(registry, consumer)` returns one PASS, FAIL or SKIP per consumed
registration. The consumer is replaceable. The first differential consumer
resolves an optional executable reference, derives bounded witnesses from the
declared `when` alternatives, executes the exact registered preparation/program, and applies a centralized
numerical policy plus exact logical-effect checks. Missing reference/target
coverage is reported explicitly; it is not a runtime admission requirement.

```bash
python -m mstar_accelerators.testing --report discovery.json
```

Discovery without model assets diagnoses one SKIP per file and initializes no
native context. An empty corpus also remains an honest SKIP. Driver configuration supplies assets
and the managed Pi execution environment; the consumer owns case selection.
`--require-all-pass` is an explicit campaign gate;
ordinary discovery does not fail simply because the current machine lacks a
reference or target. See `docs/QUALIFICATION.md` for the interfaces and commands.

## Extending the first candidate

New declarations can use new conditions and physical programs without modifying
the model's identity. Each new condition or preparation requirement needs a
corresponding consumer. The initial binder supports finite batch bounds, a
singleton progress check, per-hosted-walk partial evaluation and explicit
cost-based overlap resolution. Optional helper requirements can diagnose a
contradiction with `when` using the same condition language. A diagnosis never
silently narrows `when`, and unknown analysis imposes no proof obligation.
Workspace readiness belongs to the single `acquire(facts)` operation. This is
not a global scheduler optimizer or an autonomous factory.

Additional hardware belongs below the existing execution and memory contracts.
Additional resources belong in M*'s resource lifecycle. Additional model
semantics belong in the model. Capture, distributed role assembly, prefix reuse,
quantized numerical policies and new serving features are future consumers of
those boundaries; the first Pi declarations do not advertise them.

Package 2's native sampler uses a documented stateless counter mapping and the
existing graph stream, output and retirement paths. M* still owns canonical
policy, seeds, offsets and retained-row commit. No CPU logit readback or private
sampler is introduced. The central consumer checks bounded distributions and
exact seeded replay; it does not require every implementation to share this RNG.
Read `backends/pi/docs/categorical.md` for the primitive and `docs/QUALIFICATION.md`
for the separate acceleration policy. The distributed agreement consumes the
same execution options and resource plans. The authored CPU TP2 declarations
choose their collective through ordinary program code; neither collective
requires a separate execution kind.
