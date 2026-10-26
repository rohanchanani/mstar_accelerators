# Execution requests and physical programs

An acceleration identifies a logical transition, states when it can implement
that transition, and supplies a physical implementation. Its reference does not
select a numerical program, programming language, compilation policy, or runner.
Those are implementation choices. M* owns the mechanisms that consume those
choices: resource construction, scheduling, streams, admission, graph slots,
completion and commit.

The outer declaration remains `Acceleration(reference, when, implementation)`.
`Implementation.execution` is the execution request within its third field. The
current choices are a callable execution, represented by `execution=None`, and
an explicit `TensorForwardExecution`. Neither changes the meaning of `when` or
introduces a qualification verdict into serving.

## Callable execution

The default remains the existing `Implementation(prepare=..., resources=...)`.
Its preparation returns `PreparedImplementation(materialize, primitive,
acquire, close)`. M* selects it using the declared support and cost, acquires its
workspace, admits the model's resource effects, materializes a call from the
logical invocation and plans, then invokes `primitive(call, execution)`.

An ordinary Python callback, the optional CPU `TorchProgram` helper, and the
optional Pi ggml graph helper all use this interface. Choosing a collective,
compiler, graph language or native kernel inside such a program does not require
a new execution kind. The helper provides physical preparation and workspaces;
it does not become a request scheduler or resource owner.

## Tensor-forward execution

The author requests M*'s existing tensor forward and accelerator graph machinery
explicitly:

```python
from mstar.engine.tensor_forward import CapturePolicy, CompilePolicy
from mstar_accelerators import Implementation, TensorForwardExecution
from mstar_accelerators.backends.torch.tensor_forward import tensor_resource_requests

def prepare(binding, resources):
    return Program(binding, resources)  # An independent torch.nn.Module.

implementation = Implementation(
    prepare=prepare,
    resources=tensor_resource_requests,
    execution=TensorForwardExecution(
        compile=CompilePolicy(enabled=False),
        capture=CapturePolicy(
            enabled=False,
            required=False,
            compile=CompilePolicy(enabled=False),
        ),
        autocast_dtype=None,
    ),
)
```

`prepare` constructs the implementation's physical module, prepares its weights
and chooses its resource views. It does not install compute on the logical shell
or invoke a model-owned reference-program factory. An optional `program.close()`
releases preparation roots at the runtime's existing close boundary.

The registry packages this module and the requested policies into M*'s prepared
tensor-forward carrier. The resource requests are visible before resource
construction; preparing a module cannot retroactively change its storage or plan
requirements. `tensor_resource_requests` is an editable optional physical choice
for the current runtime numerical resource methods and device plans. Logical
resource identities, allocation, admitted rows, rollback and commit remain with
their existing owners.

### Program hooks

| Hook | Physical decision it expresses |
| --- | --- |
| `prepare_rows(inputs)` | Optional physical tensor preparation before runtime padding; preserves logical row order, walks, lengths and declared effects. |
| `preprocess(graph_walk, engine_inputs, inputs)` | Packs already-interpreted logical rows into the keyword arguments used by the forward method. |
| `forward(graph_walk, engine_inputs, **kwargs)` | Computes one request's physical output storage, using logical contract field names. It may return the same `BatchedModelOutput` carrier as the batched method. |
| `forward_batched(graph_walk, engine_inputs, **kwargs)` | Computes a batch and returns physical output storage through `BatchedModelOutput`. |
| `can_batch(batch, inputs)` | Declares whether this program can pack the proposed rows together; authored `when` still governs supported invocations. |
| `get_accelerator_graph_configs(device, tp_world_size=1)` | Supplies explicit static shapes and forward methods to capture when capture is requested. |
| `close()` | Optional release of implementation-owned preparation roots after the runtime's execution lifecycle. |

The generated `Inputs`, `Plans`, `Resources` and `Call` records still describe
the logical node boundary. A physical program can choose different packing and
call structures. The tensor-forward template makes that choice visible through
its program hooks instead of disguising physical packing as a logical model
property.

### Output conversion stays separate from semantics

The physical output uses **logical contract field names**, not graph-edge names.
`compiler.pack_tensor_result(result, contract)` converts a generated logical
result record into row-addressed physical output storage. This is useful for
simple programs and generated placeholders; a specialized program can construct
the same physical output representation directly.

Logical contracts may resolve named dimensions from each real input row and declare
conditional output presence from that row's walk and request metadata. The shared
binder checks exact resolved shapes and required outputs; implementations do not
choose those rules. Physical output fields unused by the logical row are omitted
from its typed result. Original logical inputs remain separate from prepared
physical rows.

An output may declare a nonempty tuple of allowed dtypes. The binder accepts
only those dtypes and preserves the returned tensor without a cast; generated
output storage uses the first declared dtype. Thinker token IDs allow int64
and int32, preserving the existing captured and eager representations.

The runner removes padded rows, remaps capture handles and preserves result
storage before the bound result adapter runs. `compiler.bind_tensor_result`
then reconstructs the contract's typed result from the collected physical
outputs and the actual logical invocation. The model's ordinary result mapper
performs the semantic mapping to graph edges and effects. A runner choice does
not create another copy of those model semantics.

### Fixed tensor programs

`TensorForwardExecution(build=..., bind_result=None)` supports early construction
and already-named logical outputs. `build(binding)` replaces `Implementation.prepare`;
supplying both is an error. It loads and places the independent physical module
before resource allocation. The runtime then calls its `bind_resources(resources)`
when resource views exist. Assembly owns early programs until binding transfers
ownership; either failure path closes them through the ordinary owner.

`bind_result` defaults to the typed binder. Set it to `None` only when the program
already returns the established named outputs consumed by logical postprocessing.
The runtime still copies captured storage, slices packed outputs, commits effects,
and invokes the logical node's filtering and postprocessing. It does not derive
an output-shape oracle from the output being checked. The central qualification
consumer continues comparing outputs and state against the model contract.

Assembly can lower a single such TensorForward program into the common tensor
lifecycle. It requires a finite deployment admission bound, all hosted walks,
TP1/SP1, the logical node’s `execution_result_is_output_only` promise,
and a condition that specializes to constant YES for every admitted batch
size while all request policy, token-length and walk facts remain unspecified.
UNKNOWN is not proof. Capabilities and registration identity remain attached to
the node; outside the proved domain the ordinary selection path remains available.
Primitive executions and alternative/conditional TensorForward programs retain
their existing consumer. No scheduler grouping or supported domain is narrowed.

A logical contract may expose `input_preparation`, a model-owned protocol for
physical operations required during interpretation. Thinker requests embedding
lookup and media sentinel construction at its existing semantic boundaries.
The physical program's `input_preparation()` creates a row-local session; the
runtime passes it as `preparation` to the logical node, then retains its
`finish(logical_row)` result alongside the unchanged logical row. Deferral and
admission retries retain both. Computation is not installed onto the logical shell.

The fixed program uses the common whole-step autocast, warmup order, graph slots,
preprocessing, forward, commit and output collection. Authored capture sizes are
physical geometry, independent of the real-request admission bound. Eager and
captured execution must still implement the same semantics.

### Compilation and capture are explicit choices

The ordinary forward compilation policy and the compilation policy used while
preparing captured regions are separate. `CompilePolicy` exposes enablement,
mode, full-graph and dynamic-shape choices, and platform overrides.
`fallback_to_eager` handles compile-wrapper construction failure only; it never
retries a failed lazy forward after effects may have occurred. Autocast is
also explicit; `autocast_dtype=None` disables ambient autocast for this program.

`CapturePolicy.enabled` requests capture of the program's supplied recipes.
`required` makes failure to establish the requested capture a bring-up failure
instead of silently presenting an uncaptured execution as that implementation.
The runtime still owns warmup ordering, stable resource buffers, graph pools,
slots, padding, plan installation and replay. A graph loop alone does not enable
capture. A supported eager prefill and a captured decode can belong to the same
program, with the actual executed path reported by the engine.

Changing program code after capture does not change an already captured graph.
Bind and prepare the chosen implementation before runtime warmup. A fresh
process and fresh evidence are required when comparing changed physical programs.

## Generate the actual authoring surface

Use the installed command and select the execution template explicitly:

```bash
mstar-accelerators create --deployment /absolute/path/to/deployment.yaml \
  --out-dir /absolute/path/to/new-books --execution callable

mstar-accelerators create --deployment /absolute/path/to/qwen3-omni.yaml \
  --nodes Talker --out-dir /absolute/path/to/new-talker --execution tensor-forward
```

The default is `callable`, independently of the deployment device. Node selection
defaults to `runtime_extension.nodes` when present; `--nodes` overrides it. The
CLI emits one source file per selected logical node, preserving correlated
walk/TP/SP support. It reads the logical model description without loading
weights or device kernels, and refuses to overwrite an existing book.

Both templates emit executable zero placeholders. The tensor-forward placeholder
has an independent program, logical-field output conversion, explicit resource
requests, disabled compilation and disabled capture. Its empty capture-recipe
hook is an authoring decision to fill before enabling capture. No field labels
the book unfinished or qualified. The central testing consumer assesses the
actual implementation.

Preserve the exact generated files outside corpus discovery, then retain the
generated-to-filled diff. A substantive implementation-interface change requires
the generator, assembly/runtime consumer and qualification consumer to change
together. A handwritten filled book alone cannot demonstrate that the authoring
surface exposes the new choices.

## AMD implementation and qualification

The AMD deployment selects Talker, Thinker and the audio encoder through
`mstar_accelerators`. Their physical programs are owned by this repository;
vision and Code2Wav retain their existing deployment paths. See `QWEN3_OMNI_AMD.md` for the concrete
hardware, environment and service checks.

The qualification consumer must execute the selected prepared program through
the actual runner boundary. An optional portable numerical oracle is a separate
testing choice. Missing model-case coverage or an unavailable target is an
explicit SKIP; an execution failure is not reference unavailability. CPU
arithmetic and replay-control tests verify their stated mechanisms. They do not
establish that Qwen weights, ROCm kernels, a captured AMD decode or a complete
service request ran successfully.
