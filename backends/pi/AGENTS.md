# Working in the Pi backend

Read README.md, docs/CONTRACT.md, docs/PROVIDER.md and
docs/managed_graph_provider.md before changing behavior. Use the current
delivery's CODEX_RECIPE.md for execution. This source extends the accepted
Stage 2 native/Torch/ggml integration with a complete graph provider. Model
programs live in the external library's implementation directory. This backend
is not the previous Pi node factory or an independent model server.

Keep model semantics, deployment policy, acceleration programs, and hardware
execution ownership separate. M* remains the serving runtime. This package
implements the hardware/Torch boundary; it does not import M* or own requests.

Prefer straight-line, explicit ownership and cohesive files. Extract helpers
for independent responsibilities, not to shorten functions. Unsupported
operations must reject before submission. Never add an implicit CPU fallback,
dummy tensor operation, global synchronization workaround, special-case model
name, or weakened oracle to make a test pass.

Read the complete failing path before changing it. Completion, successful
execution, visibility, storage lifetime, and live-subrange reuse are distinct.
Errors and timeouts are not cancellation. An error-signaled fence is not by
itself permission to recycle storage. A reset invalidates stateful execution.

The private kernel contract in kernel/ is required for native submission.
Do not bypass its capability query with an environment flag or version string.
Do not install a kernel patch before it applies, builds, and its source-level
validation passes. Never unload a poisoned driver; recovery is a controlled
reboot with provenance retained.

Use tools/run_hardware.py for Pi 5 execution. It uses the established shared
lease and boot quarantine directory; do not create a parallel lock or clear
markers. Do not kill timed-out GPU work and then treat process exit as device
completion. VC4 and its active compiler project are outside this batch.

The user explicitly authorized Codex to preserve failures, diagnose them,
repair source when warranted, and continue. This includes semantic repairs
supported by the established ownership and execution contract. Keep an immutable
baseline, every failed run, the diagnosis, and a separate fix commit. Do not
stop merely because a repair is semantic. If the repair would change the core
ownership contract, require an unsupported hardware assumption, weaken an
oracle, or rely on an untracked wait/copy/lifetime, stop with a concrete finding
for the architect. Historical Stage 1 instructions do not override this rule.

Report source review, host compilation, host semantic tests, target compilation,
GPU correctness, M* integration, and performance separately. Preserve exact
versions, commands, exit statuses, and raw results. No stage is a substitute for
another. Keep this repository clean; place generated runs outside source.
