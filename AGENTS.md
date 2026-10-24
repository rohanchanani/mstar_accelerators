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

Qualification policy belongs to this optimization platform. The user explicitly
authorized the centralized registry harness and delegated numerical policy;
this delivery defines component-rms-v1. This supersedes the historical requirement
to keep the first Whisper pointwise comparator as the acceptance rule; keep its
old FAIL receipts and named diagnostic intact. Do not silently tune per-fixture
thresholds or reinterpret old evidence. Exact token/state/lifetime assertions
and the backend's original transfer/linear/graph conformance gates are unchanged.
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

The qualification harness consumes every acceleration and returns one PASS,
FAIL or SKIP per registration. Sampling, reference interpretation and numerical
policy belong to its centralized consumers. Do not add per-acceleration test
commands, driver mode filters or recipe-maintained test-case lists. Broader
coverage extends the consumer/constraint solver, using the declaration's facts.
Backend conformance and compiler unit tests test their own mechanisms; they are
not alternate acceleration qualification policies. Ad hoc investigative tests
belong in scratch with explicit evidence scope.
