# Qwen3-Omni foundation A on AMD

A serves the full Qwen3-Omni graph on one gfx942 GPU. Its Talker acceleration
owns an independent physical program and explicitly requests M*'s tensor-forward
execution machinery. The remaining four Qwen nodes retain their existing model
implementations. M* owns resource state, scheduling, admission, streams, capture
storage, output lifetime and commit; the acceleration owns its numerical program,
weight preparation, packing and compile/capture choices.

The physical starting point comes from AMD donor commit
`678f59d6adb3d892263f2ba30fcf7b28145abf2e`, adapted to the Package 3 interface.
This is A. Residual-code embedding/record/accumulation fusion, QK normalization
plus RoPE plus KV-write fusion, and batched/captured text projection remain
subsequent work for B on the same logical transition.
A reproduces the starting program's execution behavior because that is the
experiment's baseline, not because its logical reference chooses a runner.

## One command on the AMD host

Use the paired source checkouts and a Linux x86_64 host with Python 3.12, `uv`,
`git`, `curl`, a C++ compiler, a Rust toolchain, and the ROCm 7.2 driver/toolkit.
The current deployment targets an MI300X or MI325X (`gfx942`) visible as GPU 0.
Preflight requires at least 90 GiB of device memory. Allow at least 32 GiB of
host RAM for the central FP32 numerical oracle and checkpoint loading, and at
least 100 GiB of free local disk for the checkpoint, environment and caches.

From the mstar_accelerators checkout:

```bash
bash tools/qwen3_omni/run_a.sh \
  ../mstar /absolute/path/to/amd-work /opt/rocm-7.2.0
```

The command installs the pinned environment, sources its environment file, runs
target preflight, downloads or reuses and verifies the checkpoint, runs central
Talker qualification, then starts the real service and records the demo. Each
campaign has a fresh `amd-work/runs/<UTC timestamp>/` directory. A failed stage
stops the campaign before later stages run; the logs and `foundation.json` retain
the actual failure. An unavailable ROCm target is recorded as `BLOCKED`.

An existing copy of the same checkpoint can be supplied:

```bash
bash tools/qwen3_omni/run_a.sh \
  /absolute/path/to/mstar /absolute/path/to/amd-work /opt/rocm-7.2.0 \
  --checkpoint /absolute/path/to/pinned-snapshot --offline
```

`--offline` applies to checkpoint access; environment installation still needs
its package sources. Hugging Face's normal token environment or saved login is
used when required. No token is embedded in a command, deployment or receipt.
The script never installs or replaces the host GPU driver.

For another campaign using the already installed environment:

```bash
source /absolute/path/to/amd-work/environment.sh
python tools/qwen3_omni/run_foundation.py \
  --mstar-root /absolute/path/to/mstar --work-dir /absolute/path/to/amd-work
```

`--output` chooses a new run directory; existing directories are refused.
`--port`, `--startup-timeout` and `--shutdown-timeout` control service startup and
shutdown. Compilation and Hugging Face caches remain in the persistent work
directory and are reused by later campaigns.

## What the gates establish

Preflight checks the pinned packages, the actual ROCm device, source imports and
HIP GEMM, then executes focused attention, norm, RoPE, sampling, MoE and graph
replay tests. The real-device tensor-forward consumer test verifies independent
program selection, padded capture, stable logical output mapping, result lifetime
and resource commits. Every selected target test must execute and pass. The
standalone `preflight.py --imports-only` option records its reduced scope; the
one-command campaign does not use it.

Asset preparation pins checkpoint revision
`26291f793822fb6be9555850f06dfe95f2d7e695` and verifies all fifteen shards and the
metadata files against `requirements/qwen3-omni-assets.json`. It writes absolute
checkpoint and acceleration paths into the deployment. Target preflight runs
before this large download or verification pass, so an unsuitable host stops
first.

Central qualification uses the ordinary registry-consumption CLI:

```bash
python -m mstar_accelerators.testing \
  --registry /absolute/path/to/mstar_accelerators/accelerations/qwen3_omni/amd_a \
  --driver mstar_accelerators.testing.qwen3_omni:run \
  --config /absolute/path/to/run/qualification-config.json \
  --report /absolute/path/to/run/qualification.json --require-all-pass
```

The launch script supplies checkpoint, deployment and device/environment facts.
Case selection, numerical policy, state checks and captured-decode checks belong
to the central consumer. It exercises the exact registered program through M*'s
runner and compares against its independent CPU oracle. The current numerical
policy covers finite short **greedy** trajectories through all three Talker walks
and all sixteen codec groups. It is not full stochastic-distribution or whole-model
quality qualification. Its report retains each
trajectory's actual selected implementation, walk, rows and eager/captured path.
The launcher requires a complete PASS before starting service. Qualification is
not a field in the acceleration or a gate inside M* serving.

The service run uses one startup, image and audio understanding requests, one
checked speech warmup, and the same eight speech prompts at client concurrency 1
and 8. It uses M*'s existing streaming speech benchmark. The report checks the
requested concurrency and request counts, completed responses, nonempty audio,
an actual `runner=tensor_forward` captured Talker decode receipt, and matching
worker initialization/release receipts after ordinary shutdown.

The outputs include response text, every WAV, raw request timings, benchmark
commands, source identities and untracked source hashes, rendered configuration,
server logs and `demo/run.json`. `foundation.json` links the central report and
summarizes the speech results. Reported metrics are median TTFA, generated audio
seconds per wall second, and mean real-time factor. Eight prompts are a bring-up
sample, not a tail-latency or quality-equivalence study. The speech is Qwen's
spoken answer to the prompt, not a verbatim reading. Inspect the media responses
and listen to the saved audio before presenting it.

Source identity works for both supported delivery forms. A Git checkout records
its own HEAD, working diff and untracked source hashes. A source archive records
deterministic regular-file hashes and a tree digest, excluding generated caches,
build directories and credential-shaped files; an adjacent export manifest is
recorded when present. No synthetic Git history is required.

## Pinned environment and startup

The retained recipe pins Torch 2.12.1+rocm7.2, Torchvision 0.27.1+rocm7.2,
Torchaudio 2.11.0+rocm7.2, Triton-ROCm 3.7.1, AITER 0.1.25, FlyDSL 0.3.4.1,
AMD FlashInfer 0.6.18+amd.1, pybind11 3.0.4, Transformers 5.10.2 and NumPy 2.2.6.
The AITER wheel is checked against its committed SHA256. AMD FlashInfer is
fetched at commit `7e2913df7604a647db40ed44dac6bb23575cc838`. M*'s Rust extension
is built in release mode from the paired checkout. `environment-freeze.txt` and
the timestamped bootstrap log retain the installation result.

Cold ROCm compilation, especially Code2Wav's MIOpen search, can take many
minutes. The startup allowance is 5,400 seconds, and progress is retained in
`demo/server.log`. Reuse compatible caches. `MIOPEN_FIND_MODE=FAST` is rejected
because it changes the retained vocoder tuning policy; an empty `AITER_JIT_DIR`
is rejected because it hides the wheel's prebuilt kernels.

The acceleration retains the donor's eager fallback policy (`capture.required`
is false). It explicitly requests capture and the donor compilation choices;
the central consumer and demo require evidence that decode actually replayed a
captured graph. A successful startup or a capture-preparation message alone does
not satisfy those checks.

For an interactive service after a recorded campaign:

```bash
mstar serve qwen3_omni --config /absolute/path/to/run/qwen3_omni_a.yaml \
  --gpus 0 --tensor-comm-protocol SHM --port 8000
```

Always provide the rendered colocated configuration. The generic Qwen CLI
default selects a different multi-GPU deployment.

## The authoring boundary and the next implementation

`accelerations/qwen3_omni/amd_a/Talker.py` is the filled book. Its logical
reference is `mstar.model.qwen3_omni.logical.TalkerSubmodule`. Its `when` states
ROCm, BF16, TP1/SP1 and the supported walks. `Implementation.execution` contains
an explicit `TensorForwardExecution`; preparation returns the external program
from `implementations/qwen3_omni/amd_a`. No physical program is installed on the
logical shell. `DeviceTorchRuntime` only initializes the Torch GPU, assembles the
ordinary registry, drains work and releases its roots. It does not choose the
programming model or runner.

To expose the same current implementation interface for another book:

```bash
mstar-accelerators create \
  --deployment /absolute/path/to/run/qwen3_omni_a.yaml --nodes Talker \
  --execution tensor-forward --out-dir /absolute/path/to/new-talker
```

Preserve that exact generated source, then fill its physical program and policy.
The generated typed logical records and the model's output mapper remain the
semantic boundary. For B, the physical predictor prologue, residual-code
embedding/accumulation and raw-conditioning projection are the intended edit
points. Keep the experiment's sampling policy, precision, resource effects and
audio chunking comparable. Select the new implementation before fresh process
startup and capture. The execution-request details are in
`EXECUTION_IMPLEMENTATIONS.md`.

## Host evidence has a distinct scope

Configuration-only generation works without weights, a GPU or Triton. CPU
runner tests, replay-control tests, independent small-model trajectories,
checkpoint metadata checks and offline target compilation establish their
specific boundaries. They do not establish live ROCm execution, full-model
memory headroom, device graph replay, audio quality or speed. The target
campaign above produces those execution receipts when suitable hardware is
available. The production A declaration has no CPU serving fallback.
