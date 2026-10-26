# Qwen3-Omni on AMD MI300X

This is the single retained AMD implementation used by the measured service.
It serves the full five-node graph with BF16 parameters, TP1/SP1, ordinary
sampling, and optional captured execution. Eager execution implements the same
semantics. MStar owns scheduling, request state, allocation and output lifetime;
the Talker acceleration owns its physical program and weight preparation.

## Install and prepare

Use paired `mstar` and `mstar_accelerators` checkouts on a Linux x86_64 ROCm
7.2 host with Python 3.12, Rust and a C++ compiler. The pinned dependencies and
checkpoint manifest are in `requirements/`. The checkpoint is
`Qwen/Qwen3-Omni-30B-A3B-Instruct`, revision
`26291f793822fb6be9555850f06dfe95f2d7e695`.

From the accelerator checkout:

```bash
bash tools/qwen3_omni/bootstrap_rocm.sh ../mstar /absolute/path/to/amd-work /opt/rocm-7.2.0
source /absolute/path/to/amd-work/environment.sh
python tools/qwen3_omni/prepare_assets.py --config /absolute/path/to/amd-work/qwen.yaml
```

Asset preparation downloads/verifies the checkpoint and renders absolute paths
from `deployments/qwen3_omni_amd.yaml`. It accepts `--checkpoint` and `--offline`
for an existing pinned snapshot. The bootstrap does not install GPU drivers.

## Serve

```bash
export OMP_NUM_THREADS=4
ulimit -n 65536
python -m mstar.cli.main serve qwen3_omni \
  --config /absolute/path/to/amd-work/qwen.yaml --gpus 0 \
  --host 127.0.0.1 --port 8000 --tensor-comm-protocol SHM
```

Use the supplied colocated deployment, not the generic multi-GPU default.
The measured deployment uses `max_concurrent_requests: 8`; benchmark client
concurrency can exceed service admission concurrency. Do not change that setting
when reproducing the measured configuration.

## Implementation and checks

`accelerations/qwen3_omni/amd/Talker.py` declares the Talker implementation.
Its physical program and kernels live in
`src/mstar_accelerators/implementations/qwen3_omni/amd/`.
The paired runtime contains the selected Thinker, sampling and MoE optimizations.
Source attribution is retained in that implementation's `PROVENANCE.md`.

`bash tools/qwen3_omni/run_amd.sh ../mstar /absolute/path/to/amd-work /opt/rocm-7.2.0`
runs the existing target preflight, qualification and service checks. These
require the GPU and checkpoint; CPU checks are not substitutes for GPU evidence.
The source-layout cleanup does not itself constitute a new performance run.
