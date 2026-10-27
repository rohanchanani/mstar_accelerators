# Ownership and execution

## 1. The reference is a state transition

The semantic object remains `R[W,C](I,S) -> (O,S',E)`: logical weights and
configuration, invocation input, persistent state and relevant randomness,
outputs, successor state and observable effects. A model's graph and walks
compose those transitions. The first implementation uses the existing
`NodeSubmodule` identity; it does not introduce a second `NodeDefinition`.

The M* model supplies a fresh configuration/lifecycle shell through
`Model.create_submodule`. `Model.logical_weights` exposes `LogicalWeightSource`
identities, shapes, aliases, defaults and values before deployment-specific
packing or sharding. These descriptions can be imported and generated without
loading a physical program or device kernels. `make_reference_program` explicitly
constructs an optional physical oracle for qualification. It is not an
implementation-construction hook for serving an acceleration.

Whisper's retained input interpretation produces logical audio features, token
IDs and encoder states. Its retained `declare_step` owns KV, cross-KV, position
and sampling effects. Reference-specific physical `preprocess` is bypassed only
for a bound execution; its work is accounted for by the implementation's
materialization and complete program. External graph edges and walks do not
change when a program fuses internal work.

## 2. Three declaration fields, one derived assembly

`Acceleration` contains only `reference`, `when` and `implementation`.

- `reference` is the model-owned submodule type.
- `when` is a typed condition with `evaluate` and `specialize`. Missing facts
  stay `UNKNOWN`. Static device, dtype, participant and geometry facts are
  consumed at binding; walk, batch and interpreted token counts remain dynamic.
- `implementation` owns preparation, an execution request, typed physical
  resource-view requests and an optional cost. The execution request says how
  M* should consume the prepared physical program. Callable preparation supplies
  materialization, primitive, workspace acquisition and close callbacks.
  Tensor-forward preparation supplies an independent physical module; its
  execution request chooses the runtime's compile/capture behavior. Prepared
  constants and programs remain implementation-owned. The runtime retains the
  reservation and completion machinery, including `acquire(facts)` for callable
  workspace acquisition.

`Registry.assemble(NodeBinding)` derives `NodeAssembly`. This is a runtime
product, not another handwritten model specification. M* collects typed physical requests and constructs early tensor programs when
requested, resolves and builds its ordinary resources, then binds programs and
installs `ExecutionCapabilities` on the fresh shell. A provably fixed named-output
tensor program uses the common tensor lifecycle; conditional programs retain
per-invocation selection. See `EXECUTION_IMPLEMENTATIONS.md`. Program code
does not replace methods or install parameters on that logical shell. The
runtime binds the selected physical program before caching runner callables.

The filesystem CLI reads `Model.describe` and the existing deployment YAML. It
writes one source file per logical node, with generated typed `Inputs`, logical
`Plans`, `Resources` identities and `Call` records. Its explicit execution
template exposes the corresponding preparation and runner decisions. The default
is callable and does not depend on the device. The initial `when` preserves
the deployment's correlated walk/TP/SP cases and identifies the logical node;
two nodes can use the same submodule class without their generated files
colliding. An unspecified deployment target stays unspecified in the scaffold.

For the Pi example, one encoder declaration and one decoder declaration bind
into two submodules. The decoder's two internal programs share prepared weights
and persistent resources. Explicitly copying and specializing its source adds
implementation choices to that same node. Copies need no identifier rewrite or
registration edit. The compiler specializes each candidate separately for each
hosted walk before computing bounds, collecting physical requests or preparing
programs. It passes that local walk set to preparation. A callback remains
ordinary code; it can use this scope without repeating placement interpretation.
The optional graph helper prunes its mode mapping automatically.

The default compiler derives input/result materialization from the model's
`logical_contract`, and logical resource-use handles from admitted plans. A
`SequenceUse` names the admitted sequence use; `AttentionUse` names logical
queries and contexts. An implementation selects `resource.view(request)` during
preparation and uses that view to obtain physical addresses for each call. The
resource owner retains allocation, admission, rollback and commit ownership.

The existing graph adapter is an optional implementation helper. The CLI,
registry and backend runtime know nothing about ggml. A caller selects a provider
explicitly with `prepare_graph_programs(..., provider=...)`. Its graph-builder
convention is `build(*, graph=None)`: native preparation calls `build()`, and the
explicit host interpreter injects `graph=...` by keyword. The latter must not
call the keyword-only builder positionally. A different compiler or native
program can supply the same prepared callback shape without using this helper.
The helper's `input_bindings` hook permits another physical port convention;
it does not introduce another resource or request-state owner.

For contiguous access, bind `dense_views(binding, resources)` once and supply
`partial(dense_graph_bindings, views=views)` as the graph helper's input binder.
It shares logical input/position staging with the paged adapter. Its ports are:

| Port | Torch representation |
| --- | --- |
| `<kv>.<label>.keys.<layer>` / `values.<layer>` | Direct `[capacity, local_heads * head_dim]` storage aliases |
| `<kv>.<label>.write.positions` | The owner's admitted append positions, int64 |
| `<attn>.<label>.read.keys.<layer>` / `values.<layer>` | Direct context aliases in query order |
| `<attn>.<label>.read.mask` | `[queries, capacity]` validity/causal mask |

The graph author chooses these ports and supplies the computation. A cache
write's returned alias must feed its later reads so the graph records the
dependency; supplying another input alias does not establish that edge. Neither
adapter chooses a cache owner, advances a counter, or copies cache contents.

Package 2 adds an optional `sampling=True` input-adapter choice. It derives
`<sampler>.seeds` and `<sampler>.offsets` as I64 row vectors and
`<sampler>.temperatures` as F32 from the existing immutable `SamplingPlan`, in
the owner's row order. Greedy bindings omit these transfers. The optional
`categorical_tokens` helper owns this physical port spelling; it has no sampler
state. The retained decoder builders accept an implementation-selected
`sample_tokens` callback after model suppression; the default remains argmax.
Nothing about this ggml convenience enters generated declarations or generic
runtime preparation services. An optimizer can supply another realization.

A preparation scope may check a known requirement against the local authored
claim, using the same `Eq`, `In`, `Range`, `All` and `Or` language as `when`.
A concrete contradiction is a compiler error with its witness and origin. No
witness means unresolved analysis, not proof. Opaque callbacks need no extra
annotations. These checks never intersect implementation restrictions into the
serving predicate. Deployment admission bounds are legitimate independent
restrictions; a testing fixture's finite domain is not.

The optional ordinary-Torch helper uses the same `PreparedImplementation`
boundary. `TorchProgram` callbacks consume generated calls and return logical
output fields; the helper copies into newly owned result storage and provides
atomic workspace leases. The explicit CPU runtime loads the registry without
choosing a graph language, interpreting Pi programs, or constructing references.
M* supplies the actual participant groups and owns distributed agreement.

Logical-weight conversion now lives in `backends/weights.py`, shared by the
optional Torch and ggml paths. A `TensorShard` selects one logical dimension;
rank/count come from the binding. Model-owned checkpoint interpretation and
aliases remain unchanged. The bounded CPU realization uses FP64 prepared
weights/arithmetic and the original FP32 output/cache interfaces. Its authored
FFN callback chooses the collective and adds the second bias after reduction.
See `TORCH_IMPLEMENTATIONS.md` for the concrete ownership boundary.

## 3. Support, scheduling and resource ownership

M* owns `ExecutionCapabilities`; the external DSL compiles into it. The scheduler
queries possible executions for a proposed batch. Its initial bounded FIFO
search considers complete candidate sizes, so exact-two support does not imply
singleton support or require an executable singleton prefix.

The engine retains logical input rows while refining a scheduling proposal,
then reserves a prepared program slot before resource admission. If a proposed
mixture is unsupported, it searches supported subproposals rather than failing
individually valid requests or repeating effectful input preparation.
Supported-but-busy returns a retryable admission outcome. Overlapping
supported programs need comparable costs; equal available minima are an error.
Neither registration order nor exception retry is an implicit selection policy.

Complete invocation facts carry ordered `row_walks`. A uniform proposal uses
its actual walk; a genuinely mixed proposal retains the combined invocation
label and its row composition. Authored support decides whether that mixture
can run before the scheduler considers uniform alternatives. The generated
call exposes the same composition. Walk-specific inputs retain `None` for
absent mixed rows; an input absent from every row remains an empty tuple.

The acquisition callback receives invocation facts so one implementation can
select the correct mode/shape workspace pool. The binding includes the actual
walks hosted on that worker; coverage is checked there, not against walks placed
on other workers. One named node is constructed once per worker, not per walk.

The first binder requires a finite search bound (from `when` or deployment
admission) and possible singleton progress for each hosted walk. This is an intentionally small consumer, not a proof of
coverage for arbitrary predicate languages. The first Whisper domain has fixed
model-controlled Q4/Q1 inputs. General workload admission or more elaborate
batch search extends the same boundary when it is implemented.

M* keeps its ordinary resource `declare/admit/plan/commit` lifecycle. Typed view
requests choose physical representation while retaining allocation ownership,
rollback, generation checks and logical counters. The compiler rejects a request
for a resource not declared by the node. Logical sequence/attention uses, host
position snapshots and `SamplingPlan` expose the already-owned facts needed by
a native program. The implementation returns `SamplingResult`;
the runner binds that result and commits exactly once. It does not increment a
private sampling counter or allocate private logical KV pages.

The model declares logical sequence geometry through `SequenceKVConfig`.
Deployment supplies byte budgets and policies. Physical resolution happens only
after collecting implementation requests. `PagedSequenceView()` leaves the page
size open; `PagedSequenceView(page_size=...)` requires that size.
`DenseSequenceView()` requests one contiguous block per sequence. A flexible
paged view and a dense view can share one owner backed by full-sequence blocks.
Conflicting fixed page sizes or partition requests are rejected; there is no
implicit state migration. The initial dense path is local, resident and unshared.
Page counts are derived from geometry, dtype and budget, not model constants or
request count. Existing physical configurations retain their ordinary path.

Sampler ownership is neutral to the implementation. Its canonical normalized
policy rows flow into invocation facts under the resource identity. For example,
`AllRows("sampler", Eq("temperature", 0.0))` is a greedy implementation's authored
support condition. The compiled union controls admission; another realization
can support other policies without replacing the common sampler owner.
Admission uses the known request policy with unknown walk and batch shape.
Only a conclusive contradiction rejects there; complete invocation selection
uses actual policy rows in the same order as the inputs.

In this logical resource path, a storage override cannot rewrite the model's
sequence bound. The dense resource owner checks the projected sequence length
against its resolved capacity during admission. A future implementation with a
smaller physical context capacity needs an explicit requirement consumed by the
resource owner's admission logic, including forks and prefix adoption. It must
not maintain a parallel sequence-length counter or
pretend a sizing field implements that policy. The current fixed-padded graph
uses the complete logical bound and rejects out-of-range views before submission.

## 4. Calls, workspace, results and completion

For a callable execution the primitive is `primitive(call, execution) -> result`.

- `call` is implementation-owned materialization of the logical invocation and
  its bound resource plans.
- `execution.stream` is the worker's actual current Torch stream.
- `execution.workspace` is the selected implementation's acquired lease.
- The typed result carries output tensors and model-specific physical results.
  The shell adapts it to `ExecutionResult(BatchedModelOutput, resource_results)`.

A returned Tensor owns its ordinary Torch Storage. The native provider retains
all allocation uses independently of Python owners until physical completion.
Scratch arenas remain leased by accepted work. Returning a scratch view does
not turn it into a durable result; the first programs write their results to
fresh caller-owned tensors and write persistent KV only through planned views.

The CPU combined decoder shares one `WorkspaceCapacity` across prefill and decode
programs because they use the same native scratch. Each program tracks its own
leases; closing it cannot close a sibling's capacity. The last prepared-body
reference closes the shared capacity before destroying the native handle, then
drops borrowed storage references only after native destruction succeeds. A busy
or failed close retains ownership for drain/retry; it is not permission to free
buffers.

The Pi backend joins provider work into the current Torch stream before return.
The worker records and consumes its existing Torch event, stages CPU outputs
where required and observes them only at the established read boundary. This is
asynchronous submission under M*'s execution model. It is not a claim that M*
never waits, that two physical engines overlap, or that every graph edge stays
on device. The transport and stop-processing paths remain ordinary M* machinery.

Prepared ggml graphs are implementation preparation and do not request Torch
accelerator graph capture. Tensor-forward is a separate execution request: the
acceleration prepares its own physical module and explicitly chooses compilation
and capture policy. M* consumes that request using its existing graph buffers,
slots, warmup, resource plans and replay machinery. A model graph loop alone does
not request a capture. The logical shell still owns request interpretation,
declared effects and output semantics; it is not used as a container for the
external compute program. See
[EXECUTION_IMPLEMENTATIONS.md](EXECUTION_IMPLEMENTATIONS.md).

## 5. Hardware and process lifetime

The native Pi context is the authority for allocations, queues, dependencies,
visibility, accepted submissions, immutable completion snapshots and retirement.
The Torch adapter translates these into `PrivateUse1` tensors, retained
operators, streams and events. The actual ggml provider imports canonical BOs
and submits through that same context. It has no independent weight/state heap.

The external runtime adapter initializes that backend in the spawned worker.
M* consumes health, drain and close at worker boundaries. Ordinary exceptions
quiesce submitters and drain before request cleanup. A poisoned or uncertain
context stops admission and holds its native roots for supervised recovery.
An error, a timeout, process exit and physical settlement remain distinct.

The existing physical-device launcher now has an optional managed process-tree
mode. A worker is reserved before spawn and acknowledges release only after
native shutdown. The same `flock` open-file description crosses the API,
conductor and worker processes. No process calls `LOCK_UN` on a shared lease.
Parent exit, an ambiguous spawn failure or missing release acknowledgement never
creates permission to reuse the device. The original boot quarantine and fixed
recovery procedure remain authoritative.

## 6. Scope and future extension points

The Pi declarations support Tiny geometry, FP32, TP1/SP1, B1, Q4/Q1, greedy
sampling, two live requests, no prefix reuse and no CPU offload. Model-controlled
stopping remains 444 generated tokens at the 448-token text boundary. These
limits are explicit conditions, resource support and deployment configuration.

The original Pi candidate did not implement Qwen, distributed TP/EP/CP,
quantization, graph capture, prefix transactions, resource conversions or
additional hardware. Subsequent CPU Torch/distributed work and the Qwen AMD
tensor-forward implementation extend that starting point. Their extension points are logical weights/state, physical view
resolution and preparation, supported invocation facts, compatible role binding and completion
contexts. No new axis can be enabled just by writing a predicate without its
consumer. This source is the first complete vertical slice of that foundation,
not a claim that all contemplated consumers already exist.

Qualification belongs to the external optimization platform. Its stable outer
identity is consumption of the registry with PASS, FAIL or SKIP results; the
platform-owned policy selects the differential consumer and reference adapters.
User configuration supplies assets and execution environments, never a driver
or acceptance/coverage overrides. The differential consumer solves the
current condition language against actual binding facts and realizable case
domains, executes the exact registered preparation, and applies centralized
numerical and exact-effect checks. Missing executable references and unsupported
environments are diagnosed SKIP. A runtime execution error still reaches the
managed owner for drain or fatal containment before another case can submit.

The executable-reference builders already supplied by the model remain useful
evidence sources. Their optional selection and evaluation precision are testing
policy; the runtime never constructs them merely to deploy an acceleration.
`Acceleration(reference, when, implementation)` is unchanged. No mandatory test
annotation, fourth semantic identity, proof or qualification gate is added to
M*. Different checks, numerical budgets, agent review, type checking and more
powerful test generation can extend this consumer through separate, justified
platform-policy changes. See
`QUALIFICATION.md` for the concrete first policies and honest scope limits.

## Automatic qualification setup

The testing entry point accepts a registry and report destination. The platform
corpus supplies pinned model configurations and fixtures; M* describes logical
nodes. Backend discovery supplies only hardware and lifecycle facts. The central
planner matches reference types and specializes support, and the central model
consumer materializes cases for execution through M*. The resulting plan is
report data, never an input interface or optimizer-authored configuration.
Missing environments, assets and unsupported case domains remain missing evidence.
