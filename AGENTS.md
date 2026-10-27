# Working on mstar_accelerators

Read README.md, docs/ARCHITECTURE.md and docs/CHANGE_MAP.md. For native changes,
also read backends/pi/AGENTS.md and its execution/provider contracts.

Keep the three declaration fields coherent: reference identifies semantics,
when states support, implementation supplies physical realization. Derive facts
already owned by the model or runtime. Do not add a fourth source of model,
deployment or state truth to make one implementation work.

M* owns serving. This repository must not grow a model server, duplicate request
scheduler, private KV allocator, second sampling lifecycle or hidden CPU compute
fallback. The explicit CPU evaluator under tests is an oracle and integration
test; it is never selected by the Pi deployment.

Read complete failing paths before changing them. Preserve failures and the
immutable delivery baseline. The user authorizes diagnosis, semantic repairs
consistent with the contracts, separate fix commits and continuation. Do not
weaken a fixture, tolerance, ownership rule, failure status or unsupported-domain
rejection. A repair requiring a different core ownership contract or an
unverified hardware capability must be brought back with a concrete reproducer.

Qualification policy belongs to this optimization platform. The user approved
a layered ordinary inference-model validation contract: one pinned PyTorch
Dynamo numerical comparator for floating outputs, centrally fixed generation
comparisons and reference speech continuations, and exact semantic safeguards.
Both reference and acceleration execute through M*. Do not reintroduce
component-RMS, cosine, migration-specific acceptance, or optimizer-authored
testing policy. The central generated-output policy is distinct from full-node
numerical equivalence; preserve numerical failures as diagnostics and never
claim generation-quality PASS establishes the former raw-logit requirement. Preserve historical failures. Runtime-owned
state, lifetime, sampling effects and backend conformance remain strict.
See `docs/QUALIFICATION.md` for the numerical contract and its scope.
Missing executable references or unavailable target environments are diagnosed
SKIP, never manufactured PASS. Actual bugs and execution failures are not
reclassified as reference unavailability. Serving consumes no test verdict.

Use focused semantic tests at the changed boundary and the delivery's regression
gates. Do not substitute mocks, metadata construction or CPU graph evaluation
for target execution evidence. Preserve generated receipts outside source.

## Coloring-book authoring and the current corpus

The four files under `accelerations/whisper/pi_v3d` are the exact returned,
qualified Package 2 encoder, greedy paged prefill, greedy contiguous decode and
combined native stochastic decoder. Keep their provenance. The three files under
`accelerations/whisper/cpu_tp2` contain the authored CPU encoder and two combined
distributed decoder variants. The encoder uses rank 0; the decoders partition
FFN work across two CPU workers while attention and KV remain replicated. Their
all-reduce and gather/local-sum choices are physical implementation decisions.

For a new book, run the environment's installed `mstar-accelerators create`
command into a fresh directory outside corpus discovery. Preserve the exact
generated files and the subsequent edit diff. Generate one file per logical node;
copy a file deliberately when authoring an alternative implementation. A runner
or programming model is an author choice, never inferred from the target device.

Keep the generated logical reference and typed records. Reuse retained math,
model suppression and existing state/lifetime owners. Explicit FP64 CPU
arithmetic still uses the generated FP32 logical outputs and persistent KV.
The implementation chooses its programming model, weight partition, collective,
views, programs, slots, runner request and support. See
`docs/TORCH_IMPLEMENTATIONS.md` and `docs/EXECUTION_IMPLEMENTATIONS.md`. This CPU
deployment is not a Pi fallback.
Do not restore the deleted Whisper program/support integration or superseded Pi
factory. When exercising a new interface, generate with that installed interface
instead of substituting a historical filled book for the generation step.

Record source-generation commands, the initial generated files, the edit diff,
validation results and genuine authoring friction. If a missing compiler/runtime
mechanism forces handwritten plumbing into an acceleration, identify the missing
owner/contract instead of hiding it in a callback. A warranted local correction
to an authored mechanism remains a separately documented foundation repair. A
change to the ownership model must return for review unless the active user task
already authorizes that specific change.

Physical program code belongs to the acceleration or its explicitly chosen
helpers. The logical reference is not a physical implementation factory. A
runner request is consumed by M*'s assembly and execution machinery; it does not
license mutation of the model's logical shell to install compute. Keep request
interpretation and result/state semantics with the logical model, and keep
resource allocation, admission, commit and completion with the runtime. A change
to this implementation language must update the generator, runtime consumer and
qualification consumer together.

## Centralized acceleration qualification

Qualification answers whether an acceleration implements its logical reference's
semantic contract when executed through M*. The same centrally owned cases and
acceptance policy apply regardless of who wrote it, whether it was migrated or
rewritten, or whether it matches a historical winner. Provenance may be recorded
for reproducibility, but must not create acceptance exceptions, alternate
migration verdicts, source-identity gates, or evidence-review prerequisites.

The qualification harness consumes every acceleration and returns one PASS,
FAIL or SKIP per registration. Sampling, reference interpretation and numerical
policy belong to its centralized consumers. Do not add per-acceleration test
commands, driver mode filters or recipe-maintained test-case lists. Broader
coverage extends the consumer/constraint solver, using the declaration's facts.
Backend conformance and compiler unit tests test their own mechanisms; they are
not alternate acceleration qualification policies. Ad hoc investigative tests
belong in scratch with explicit evidence scope.

## Investigations must overlap execution with reasoning

During debugging and root-cause investigations, launch measurements and tests asynchronously by default. Keep their process/job identity, logs and exit status so they can be checked without blocking ongoing analysis. The objective is the shortest time to an evidenced cause and correct repair, not the largest number of test runs.

Before launching a run, state the concrete unanswered question, the relevant owner/contract, what each plausible outcome would imply, and why existing results or source inspection cannot answer it. Choose the cheapest discriminating experiment; reuse saved inputs, outputs and loaded processes. Do not rerun a whole suite to answer a local question.

While a job runs, continue tracing the relevant source and assessing results already available. Assess partial and completed results immediately, even if other jobs are still running; update the hypothesis and next action from that evidence. Do not make launch-wait-poll the investigation workflow, and do not treat a running test as permission to stop thinking. Do not launch additional tests merely to appear busy. Only wait when useful independent analysis is exhausted and a specific pending result genuinely blocks the next decision.

Keep reference validity, migration preservation, numerical agreement and application quality as separate claims. Do not search for an oracle or adjust acceptance until the candidate passes. A harness correction needs an independently demonstrated defect and a separately justified central policy change; retain the old failures and state the limits of the new evidence.

## Deployment fidelity ownership

Deployment authors select required fidelity, independently of accelerations.
The numerical default remains the pinned PyTorch comparator. The approved
quality mode names a model-declared, pinned baseline; the central harness owns
metric implementation, continuation construction and acceptance. Do not infer
approximation permission from dtype or quantization, fit thresholds to candidate
failures, add optimizer-authored callbacks/policy, or waive exact execution
obligations. Qualification records the deployment context and honest coverage.
