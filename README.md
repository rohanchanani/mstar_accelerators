# mstar_accelerators

Handwritten implementations of M* graph nodes, declared as
`Acceleration(reference, when, implementation)`.
The model defines logical inputs, weights, results and state effects. The
acceleration supplies the physical program. M* owns serving, resource lifetimes,
scheduling and state commits.

## What does an optimizer write?

Start with the AMD [Talker declaration](accelerations/qwen3_omni/amd/Talker.py)
and its [preparation function](src/mstar_accelerators/implementations/qwen3_omni/amd/preparation.py).
The declaration identifies the model-owned reference, supported domain and
physical implementation. Preparation loads named logical weights, packs them
for the selected kernels and returns an independent program.
The [program](src/mstar_accelerators/implementations/qwen3_omni/amd/program.py)
contains input packing, numerical forwards and capture recipes; its kernel
helpers use ordinary Python imports.

The [Thinker](accelerations/qwen3_omni/amd/Thinker.py) and
[audio encoder](accelerations/qwen3_omni/amd/AudioEncoder.py) are separate node
implementations. They share the model's graph and runtime-owned resources.
There is no accelerator-specific server, request scheduler or sampling state.

Create a new declaration from a deployment:

```sh
mstar-accelerators create --deployment /absolute/model.yaml \
  --out-dir /absolute/new-implementation
```

The generator reads the model's logical description without loading weights.
It emits typed input/result records and an executable placeholder for each
selected node, without overwriting existing files. Fill in support and physical
computation. The default callable interface accepts arbitrary implementations;
`--execution tensor-forward` requests the optional tensor-program template.
See the [public types](src/mstar_accelerators/definition.py) and
[execution interfaces](docs/EXECUTION_IMPLEMENTATIONS.md).

## What happens to the declaration?

| Boundary | Code | Owner |
| --- | --- | --- |
| Select source and placement | [AMD deployment](deployments/qwen3_omni_amd.yaml) | Deployment |
| Specialize support and collect physical requests | [Registry](src/mstar_accelerators/registry.py) | Acceleration binder |
| Resolve shared representations and allocate state | [Resource contracts](docs/ARCHITECTURE.md) | M* runtime |
| Load and pack weights | Implementation preparation | Acceleration |
| Reserve workspace, execute, collect and commit | [Execution and lifetime](docs/EXECUTION_IMPLEMENTATIONS.md) | M* runtime |
| Interpret results and advance the graph | Logical node and model graph | Model |

Missing support facts remain unknown. Unsupported and temporarily busy are
different outcomes. Resource views expose existing owners; implementations do
not allocate private logical KV state or advance private sampler counters.

## Run and qualify

Use editable installations of paired `mstar` and `mstar_accelerators` checkouts.
The corpus, deployment templates and tools live in the checkout.

- [Qwen3-Omni on AMD MI300X](docs/QWEN3_OMNI_AMD.md): pinned dependencies,
  checkpoint preparation, ordinary M* serving and GPU checks.
- [Torch implementations](docs/TORCH_IMPLEMENTATIONS.md): callable helpers,
  physical views and distributed computation.
- [Pi backend](backends/pi/README.md): native execution/provider contracts.
- [Central qualification](docs/QUALIFICATION.md): declaration discovery,
  executable references, numerical policies and state/lifetime checks.

Qualification evaluates the same registered implementations used by serving.
Missing hardware or assets produce explicit SKIPs. Passing CPU checks does not
establish GPU correctness or performance, and serving never consumes a test
verdict. Source attribution for the AMD implementation is retained in
[PROVENANCE.md](src/mstar_accelerators/implementations/qwen3_omni/amd/PROVENANCE.md).
