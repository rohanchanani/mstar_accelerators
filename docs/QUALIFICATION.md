# Qualification is an optimization-platform consumer

## 1. Ownership and the small permanent interface

The external platform owns qualification technique, correctness policy and its
evidence. M* owns serving. The logical reference still identifies the transition
`R[W,C](I,S) -> (O,S',E)`; testing that transition is a separate engineering job.

The public entry point is `python -m mstar_accelerators.testing`. It accepts a
registry subset and a report destination, with optional `--require-all-pass`.
It accepts no configuration, driver, model, checkpoint, comparator, tolerance,
case-domain or topology argument. The execution plan is an output, never an input.

The inference procedure is:

1. Load the selected declarations' existing `reference` and `when`.
2. Describe the platform-owned pinned model corpus using M*'s model registry,
   `Model.describe`, graph declarations and `create_submodule`. Match the exact
   logical reference type, as the production registry does.
3. Discover hardware independently: devices, capabilities, topology and existing
   runtime lifecycle. Backend discovery contains no model or correctness policy.
4. Combine logical facts, hardware facts and the central coverage domain;
   specialize `when`. The existing model-aware constructors and condition solver
   produce valid invocation/history witnesses. M* resolves and owns resources.
5. Execute through M*, compare centrally, and retain the inferred placements,
   asset identities, missing coverage, solver witnesses and observations.

`testing/policy.py` retains the reference adapters and numerical comparator.
Qwen covers batches 1, 2 and 8 with three decode steps; Whisper retains its
existing declared-alternative, paired-state, lifetime and sampling checks.
Placement coverage is bounded TP1/TP2 and SP1 where hardware can host it. A new
acceleration of the supported logical model/backend needs no test registration,
callback or recipe. New logical semantics need centrally reviewed model/case
support; new hardware needs discovery/lifecycle support. Neither extension belongs
to an optimizer's declaration. Unsupported or unresolved domains stay explicit
missing evidence; this finite corpus does not prove arbitrary advertised shapes.

Candidate execution uses M*'s actual engine. Qwen retains its `EngineManager`
path. Whisper uses `Engine.load_model`, `prepare_inputs` and `exec`; the engine
constructs resources, selects/acquires execution, admits, plans, adapts results,
commits and releases. A test-side observer at the existing logical result
adaptation boundary reads pre-commit state without replacing execution or
commit. The independent reference and initial reference-produced fixture state
remain test-owned; they are not candidate execution paths.

The low-level `run_registry(accelerations, consumer)` loop remains reusable
inside platform tests. It is not a user-configurable qualification policy.
Changes to the harness, including precision accommodations and false-PASS fixes,
require separate, justified PRs owned by the optimization platform.

The first real consumer is `DifferentialConsumer(adapters, comparator)`:

1. Resolve an optional executable-reference/case adapter by logical reference.
2. Obtain actual environment/binding facts and constructible case domains.
3. Derive bounded cases from the declaration's `when` alternatives within that domain.
4. Construct and check the actual logical invocation and state.
5. Run the exact registration's `implementation.prepare` and prepared program.
6. Compare observed outputs and state/effects using the centralized comparator,
   and aggregate the cases into one verdict for the acceleration.

The executable reference is not a fourth required acceleration field. Whisper
already has model-owned `make_reference_program` builders. The new testing-owned
adapter selects those builders and CPU evaluation precision. An implementation
without a reference adapter remains deployable and is diagnosed SKIP here.

Missing references, missing test assets and unavailable target hardware produce
SKIP before physical submission. An explicitly unsupported reference can use the
adapter's `Unavailable` preflight result. Corrupt pinned assets, bugs in
reference code and failures after submission are not silently reclassified as
unavailability. Unexpected execution errors leave the normal consumption loop:
the existing managed worker drains or holds before any native cleanup/reuse.

Empty/all-skipped suites have status SKIP. A mixed PASS/SKIP suite reports its
counts and `complete=false`. The command's optional `--require-all-pass` lets a
specific campaign require complete coverage without making every unsupported
hardware environment a failing test suite.

## 2. Concrete code boundaries

| File | Responsibility |
| --- | --- |
| `catalog.py`, `sources.py` | Discover the repository's acceleration files without an independent index |
| `testing/harness.py` | Registry consumption, result aggregation, reference/case resolution |
| `testing/solve.py` | Test-time consumer of the current predicate language |
| `testing/numerics.py` | Centralized numerical and exact-value policy |
| `testing/sampling.py` | Bounded categorical support, distribution and seed-set policy, independent of RNG algorithm |
| `testing/models/whisper.py` | Constructible Whisper cases, independent executable reference and state interpretation |
| `testing/corpus.py`, `testing/corpus_data/` | Platform-owned pinned model/configuration/fixture identities |
| `testing/assets.py` | Standard-cache resolution and checksum verification |
| `backends/discovery.py` | Model-independent hardware inventory and existing runtime factories |
| `testing/planning.py` | Logical reference matching and inferred supported placements |
| `testing/hosting.py` | Generic managed worker and actual M* communication groups |
| `testing/environments.py` | Inference orchestration and coverage aggregation |
| `testing/__main__.py` | Config-free entry point and report/exit policy |
| `testing/policy.py` | Reference resolution, coverage defaults and fixed comparator |
| `testing/runtime.py` | Case placement and read-only observations around the actual M* engine |
| `testing/whisper.py` | Shared input, observation, state and lifetime helpers for the centralized model adapter |

The environment coordinator has a fixed set of managed execution environments
(`pi`, CPU TP2, and `rocm`). All use the same central policy. Environment
configuration describes assets and placement; it cannot import executable code.
The CPU environment remains a bounded TP2 deployment, not a universal
distributed runner.

## 3. Test generation: actual support, limited search

The solver handles `All`, `Or`, `Eq`, `In`, integer `Range` and `Constant`, using fixed
binding facts and finite reusable adapter domains. Opaque predicates are tested
on actual facts/candidates; their inverse is not invented. Missing domains or an
exhausted search budget produce UNRESOLVED/SKIP. A supported witness must satisfy
the original condition's `evaluate`, and the adapter checks its materialized
invocation again before preparation.

An adapter may add relational case constraints. For example, Whisper decode
requires one token while prefill uses the model-derived prompt span. This avoids
claiming that every Cartesian product of otherwise valid facts is constructible.
The relation belongs to test-case construction, not the serving DSL.

The consumer's current `declared-alternatives-v1` policy lazily expands declared
`All`/`Or` combinations and asks for one satisfying witness per alternative. It
does not contain a Whisper, prefill or decode special case. An acceleration
covering both paired mode/shape alternatives is consumed once; the resulting
report contains both witnesses and one aggregate verdict. Adapter constraints
restrict each solve without becoming another source of serving requirements.

The policy limits branch enumeration and each solve. Truncation, unresolved
alternatives, or an executed case with no comparisons cannot produce a complete
PASS; they retain available evidence and produce SKIP unless an observed
comparison already establishes FAIL. UNSAT is relative to the current binding
and constructible adapter domain. It is recorded, not treated as a proof about
other environments. Numerical failures accumulate; execution exceptions stop
consumption and leave cleanup to the existing managed owner.

This is bounded declared-alternative sampling, not an exhaustive shape sweep,
deployment coverage, broad fuzzing or proof. New predicate nodes need generation
support or a finite domain. More systematic sampling and distribution-aware
cases belong in this same centralized consumer/solver boundary. There is no
driver option or recipe-maintained list of mode-specific qualification tests.

The Whisper adapter additionally consumes the same corpus as context for a
bounded state-transition policy, `one-compatible-prefill-per-decode-witness-v2`.
For a solved decode witness it considers other registrations of the same logical
decoder, evaluates their actual prefill facts, and samples the first supported,
representation-compatible registration by sorted name. If no distinct producer
supports the same canonical policy, the decode declaration's own prefill may
serve as the companion. That path prepares one combined capability and reports
`selection_kind=same_registration`; it does not claim a distinct pair. The order is explicit
test sampling, independent of serving costs and corpus insertion order. The
report names both registrations, records considered support/compatibility and
the union of physical requests, and identifies the resolved storage. Missing or
incompatible companions leave an explicit unexecuted trajectory record; an
independent decode PASS does not then claim a cross-implementation transition.
This samples one trajectory per decode witness, not every compatible pair in a corpus.

Compatibility uses production `Registry.assemble`, `merge_resource_requests`
and `resolve_bound_specs` before allocating state. Only the resource owner's
specific `ResourceViewConflict` is a compatibility rejection. Malformed budgets,
unknown request types and unexpected construction/execution failures propagate;
they cannot become a reference-unavailability SKIP. Neither registrations nor
models acquire an additional authored qualification field.

## 4. Numerical policy: component-rms-v1

The current default is an empirical tensor accuracy policy. For finite reference
values `b`, define `s = RMS(b)` and `d = abs(actual - b)` in FP64 analysis.

For logically FP32 output, both conditions must hold:

```
every d[i] <= 1e-3 + 1e-3 * max(abs(b[i]), s)
RMS(d)    <= 1e-5 + 1e-5 * s
```

The first bound retains a check on every coordinate and supplies a meaningful
tensor scale near zero. The second rejects a systematic bias that could satisfy
every coordinate's looser bound. Scaling uses RMS, not the largest reference
element. RMS itself is computed with magnitude scaling to avoid overflow.

These budgets describe complete-component qualification, not individual FP32
instruction accuracy. During development a tighter peak draft also rejected
ordinary HF FP32 against FP64 on the tail fixture. Independent builder/HF FP64
execution agreed to about 6.5e-12 maximum error on that fixture; the FP32
discrepancy grew through an encoder feed-forward block. The final policy uses a
clean component error budget rather than fitting a threshold to those outliers.
The aggregate budget is tighter than the draft's, and deliberate isolated
corruption and dense bias are separately rejected by tests. This evidence
supports the initial policy; it does not derive a universal error bound from
dtype or guarantee that every future correct program will pass.

Shape and actual storage dtype are exact requirements. A higher-precision oracle
does not permit an implementation to return a different storage dtype. Integer,
boolean and explicitly exact observations use equality without conversion to
float. Any NaN fails. Infinite positions and signs must match exactly; finite
scales exclude infinities, so masked logits cannot receive an infinite budget.
Empty and matching infinite-only tensors have explicit behavior.

The same centralized table has initial FP64, FP16 and BF16 profiles. These are
starting empirical policies, not claims that arbitrary models or devices have
been qualified at those precisions. The current real deployment remains FP32.
A caller may select a different comparator or `NumericalPolicy`; future strict
annotations or accuracy policies belong to that consumer. No such annotation is
required for the initial two generated declarations.

RMS scaling is not a universal semantic equivalence relation. Large-amplitude
channels affect the scale, padding affects RMS, and global statistics do not
describe every important subregion. Model-aware exact sink/state checks remain
necessary. A future consumer may compare meaningful slices or enforce a task
metric. There is no claim of transitivity or proof of correctness.

### Reference choice for Whisper

The encoder uses the model-owned HF program on CPU in FP64, promoting the exact
logical FP32 input and weight values. Native storage/computation stays FP32.
Independent HF/builder FP64 agreement on the preserved fixtures supports this
reference choice. The decoder uses the original model-owned CPU FP32 program on
the same boundary input and resource plans. The host integration regression
also retains original-FP32 composed-reference token/EOS checks.

The previous `abs(a-b) <= 3e-4 + 3e-4*abs(b)` comparison against Pi HF FP32 remains
available as `legacy-pointwise-3e-4`. Its historical FAIL is never rewritten.
The earlier centralized-policy revision deliberately changed reference evaluation
and comparison policy; this source-generation revision retains that policy.
Source review and independent high-precision agreement motivate it;
neither transcript agreement nor a desired PASS is sufficient justification.

The original native capture reevaluation is labeled evidence replay, not fresh
execution. The old comparison still reports 24/full and 3/tail mismatches. The
new policy accepts those stored observations. Fresh corpus consumption is the
acceleration qualification. Stored captures and earlier native trajectories
remain separately identified historical evidence.

## 5. Real declarations and framework regressions

The ordinary decoder declaration does not export logits. The registry harness
therefore checks its real returned tokens, same-input successor KV state,
position/sampling effects, sink preservation and cleanup. It does not replace
the preparation function with a diagnostic graph and claim the same artifact.

Each independent decode case is seeded by a CPU-reference prefill through
M*'s actual resource plans. Reference-produced physical values are copied into
the planned native allocations; M* remains the page/counter owner. Case order
therefore does not establish correctness accidentally. Test-only CPU programs
exercise decode-first ordering, the actual prepare callback, wrong-token failure
and unsupported-device skipping; they are not shipped fallback implementations.

When the corpus provides a compatible prefill companion, the adapter runs a
second case through both exact registered preparation/execution callbacks. Their
representation requirements are resolved together, and they receive the same
persistent resource owners. Prefill writes state that decode directly reads and
extends. The independent CPU reference consumes the same admitted plans with
separate physical values; it owns no allocator, positions or sampling cursor.
The checks join the decode registration's existing single verdict, with explicit
trajectory prefixes and companion identity. There is no second qualification
entry point or serving binder.

For the bounded two-slot dense representation, that case also keeps two requests
live, verifies a third request's retryable resource refusal, releases one slot,
poisons its former storage with NaNs, and runs the waiting request after
reassignment. It checks full live blocks including safe tails, the other live
request, retained outputs, exact position/sampling commits, sink preservation and
final release. Free blocks have no logical value and are not numerically compared.
The untouched survivor's self/cross blocks, position and sampling offset are
compared exactly with their own synchronized pre-interference snapshots. A
separate numerical comparison retains independent reference accuracy. The
Package 2 repair closes a gap where a tiny illicit change to the other live
request could previously pass its numerical budget. A legal prefill difference
does not invalidate exact subsequent preservation; an illicit within-budget
mutation still fails. The Package 1 pre-decode cross-state repair is retained.
Other resolved capacities keep the transition case and explicitly report that
this particular lifecycle profile was inapplicable. Resource-level compiler
tests cover incompatible requirements and contiguous capacity overflow; the
ordinary API service separately exercises scheduler backpressure, cancellation,
drain and shutdown. None is a recipe-maintained list of acceleration tests.

Workspace is acquired before each real implementation's resource admission and
ordered preparation. Physical results pass through the model shell's existing
`execution_result` mapping and `StepRunner.accept_results`/`commit`. Partially
constructed resource sets remain rooted in the managed qualification session,
so a later construction failure does not release earlier submitted work before
the worker's settlement boundary. Complete invocation facts retain actual row
walks and canonical resource-qualified policy rows in both selection and receipts.

The former standalone Whisper qualification CLI is removed. Its previous
diagnostic graphs and receipts remain in accepted history; they do not define
a second qualification surface. More observations or trajectories can be added
to the centralized model/reference consumer when justified. Ad hoc investigative
tests belong in scratch. The actual API service and host engine regressions
continue to test product/framework integration at their own boundaries.

The adapter settles and closes each successfully observed case before the next;
the Pi environment host retains the managed exceptional drain/fatal boundary and final
owner release. That synchronization is at the
qualification observation/release boundary; serving stream behavior is unchanged.

### Host graph algebra and FP32 integration

`full_checkpoint_builder_algebra_v2` separates two host-test responsibilities.
The encoder graph's algebra is evaluated in FP64 on the exact logical FP32
weights and inputs, and compared with the independent model-owned HF FP64
encoder using the existing centralized FP64 profile. This read-only evaluation
does not emulate native kernels or qualify an FP64 deployment.

The original FP32 graph evaluator still runs and supplies the encoder states
used by the complete FP32 decoder trajectory. Same-input logits and KV checks,
exact token/EOS and state/effect assertions, and result-lifetime checks remain
required. Its FP32 encoder comparison against HF FP64 is retained as a named
diagnostic under unchanged `component-rms-v1`, alongside the legacy diagnostic.
A diagnostic FAIL remains FAIL; neither is a required observation of this
versioned algebra/integration gate. Actual registered Pi programs remain subject
to their required FP32 numerical and exact-effect checks.

This is an explicit change in the host gate's claim, not an arithmetic repair.
The preceding campaign remains INCOMPLETE under its original requirements, and
the AArch64 FP32 interpreter's tail-encoder comparison remains failed. A new host
gate PASS does not qualify that interpreter as a production CPU acceleration.
The previous CPU-only revision could reuse separately identified native receipts
because its production source and binary identities were unchanged. This new
source-generation/binding revision requires fresh corpus and service execution.
Report scopes and provenance separately; do not present old device observations
as fresh execution.

The source-generation rehearsal uses the new gate identity
`generated_files_engine_algebra_v1`. It retains the same numerical policies and
observations while driving the filled source files through the actual M* engine,
with an explicitly injected CPU graph provider. It projects only the Pi device
predicate for this host interpretation; it does not emulate native kernels or
qualify a CPU deployment. Binding source has changed, so prior Pi evidence does
not qualify this new revision without a fresh target run.

### Positive-temperature sampling

The same central consumer derives bounded canonical sampling-policy witnesses
from resource-qualified `when` conditions and checks each materialized invocation
against the original claim. Its first positive-temperature profile implements
disabled top-k/min-p, top-p one and no repetition penalty. These are test-domain
bounds, not hidden serving restrictions. Unsupported or unresolved policies keep
their existing diagnosed coverage limits. No extra declaration field is required.

`categorical-cdf-dkw-v1` compares 64 draws from an identical logical input/state
against temperature-scaled probabilities from the independent model reference.
Its fixed 63-bit seeds are derived under a documented qualification namespace.
It uses empirical CDF distance with nominal DKW calibration at alpha `1e-6`
(budget about `0.3367` for 64 draws). Reports include actual tokens, seeds,
temperature, distance, budget, peak probability and distinct token count. This
is a bounded statistical check with low power on peaked distributions, not proof
of a PRNG or arbitrary-input correctness. A flat-logit fixed-argmax negative
control must fail this policy.

Vocabulary, I64 result shape/dtype and exclusion of explicitly suppressed tokens
are exact. Repeating the initial seed with the same inputs/state must repeat the
actual token exactly. Numerical KV accuracy remains `component-rms-v1`; position,
sampling effects, result lifetime and preservation remain exact. The primary and
paired trajectories retain full cache comparisons. The extra distribution draws
retain tokens and exact effects without accumulating 64 full-cache snapshots.
Prepared weights/programs/resources are reused while requests are removed and
reingested through the ordinary owner. The independent decode fixture is seeded
from CPU prefill with a fixed lawful teacher-forcing token. Paired stochastic
trajectories teacher-force actual sampled outputs into the independent reference.

This policy does not require Philox, native/FlashInfer-identical draws, or exact
CPU/GPU agreement at a random decision boundary. The optional native primitive
has its own separately documented RNG mapping and backend conformance gate.
Those checks establish a different boundary from registered acceleration
qualification. Serving consumes neither verdict nor distribution policy.

## 6. Running centralized qualification

```bash
python -m mstar_accelerators.testing \
  --registry /absolute/path/to/accelerations \
  --report /absolute/path/to/qualification.json --require-all-pass
```

Omit `--registry` to discover the checkout's entire registry. Arbitrary subsets
are supported. No acceleration-specific test settings are needed or accepted.
Small pinned model descriptions and fixtures ship with the harness. Checkpoints
are resolved from the standard Hub cache using the corpus's immutable revisions;
all required bytes are checked against the manifest before execution. Use the
ordinary Hub tooling or the repository's asset preparation tools to populate
that cache. `HF_HOME` and `HF_HUB_CACHE` are ordinary asset-location environment
variables, not correctness or coverage controls. The harness does not download
tens of gigabytes merely to discover an unavailable target.

For Pi, the same command runs under its established ownership launcher:

```bash
python backends/pi/tools/run_hardware.py --managed --deadline 3600 -- \
  python -m mstar_accelerators.testing \
    --registry /absolute/path/to/accelerations \
    --report /absolute/path/to/registry.json --require-all-pass
```

Without available pinned assets or applicable hardware, the report retains
missing evidence and no physical worker is reserved. A checksum mismatch is a
failure, not an unavailable reference. Actual computation and teardown failures
remain failures. Qualification never narrows a declaration to hide a failure.

The `execution_plan` records the platform corpus/revisions, discovered devices,
logical reference matches, placement facts, generated jobs, verified assets and
unavailable prerequisites. `executions` retains worker/role observations, the
existing case domains and solver witnesses, references and comparison results.
A passing placement cannot erase a failed or missing applicable placement.
No external plan is parsed back into the harness.

## 7. Actual runtime execution across backends and roles

The generic host receives an inferred logical-model placement. It initializes
the existing backend runtime, creates M*'s communication groups, and delegates
to the central logical-model consumer. It does not choose a model, case,
reference or numerical policy. Each implementation chooses its programming
model and optional graph provider; the host does not inject one.

### CPU TP2 state and execution

The inferred CPU TP1 encoder placement uses one spawned process; the TP2
decoder placement uses two. These placements follow logical reference matching
and `when`, not encoder/decoder rules in the backend. It initializes the same M* CPU
groups used by serving and binds the actual registered preparation callbacks.
Workspace selection and resource admission use `ExecutionAgreement`, including
rollback on peer refusal. Resolved resource geometry and candidate identities
are compared before allocation/preparation. The independent reference always
uses a trivial group and full logical weights; it never participates in the
implementation's collective protocol.

The authored CPU implementation chooses FP64 intermediate arithmetic and weight
representations. The logical contract, encoder boundary and persistent KV stay
FP32. The existing encoder FP64 reference, decoder FP32 reference and unchanged
`component-rms-v1` policy apply to those outputs and state. This does not qualify
the historical CPU FP32 graph interpreter, whose named diagnostics are retained.

The existing independent and companion cases run on each participating role,
including two-live refusal/reassignment, poisoned-tail initialization, exact
survivor preservation, retained outputs and owner effects. Exact replicated-state
and output checks additionally compare hashes of the already-observed CPU values
across roles. This is a read boundary in qualification, not a serving copy or
another state owner. The companion policy can select the other collective
registration; it remains bounded one-companion sampling rather than all pairs.

Successful CPU cases close their prepared programs and state, then normally
destroy their settled process groups. A role execution failure aborts further
consumption, records the failure and stops the other CPU-only role processes.
Failed collective groups do not enter another protocol or an ordinary teardown
collective. CPU process termination does not establish completion or release for
a native Pi owner; the native environment retains its separate existing rules.

### Ordinary service and controlled failure evidence

The central environment is not the ordinary M* serving demonstration. The same
seven greedy API cases run through the actual CPU API/conductor/spawned-workers
and SHM transport, with both rank-local execution identities and normal group
teardown recorded. The native deployment retains all fourteen mixed-policy
service cases.

The service harness can instrument an explicit CPU test deployment with one
rank-local workspace refusal, or one failure after a real decoder primitive has
executed. The refusal must release the successful peer reservation and select a
common alternate registration. The fatal control must reach the conductor and
stop both workers without local retry. These focused controls wrap execution
options for testing; they do not change the normal deployment, authored math,
selection owner, reference policy or qualification entry point.

The conductor's ordinary request-failure barrier waits for read acknowledgements,
dispatches removal, then publishes failure. Cleanup completion is separately
observed in receipts; there is no extra per-request cleanup-completion ACK in
that publication barrier. Instance-fatal notification belongs to deployment
shutdown and is distinct from that ordinary request drain path.
