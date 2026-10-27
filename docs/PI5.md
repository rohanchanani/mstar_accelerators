# Whisper on Pi 5

Run from the external repository root, in the accepted Pi environment with the
M* revision in [MSTAR_REVISION](../MSTAR_REVISION) and this repository installed editable. The merged environment
uses AArch64 Python 3.13.5, Torch 2.10.0+cpu and a matching native-extension ABI.
The preserved Torch 2.9.1 control remains separate; target qualification of the
merged environment must complete before attributing the earlier measurements to it.
Audio-file input also needs [TorchCodec and FFmpeg](../tools/environment/README.md).
Reuse verified dependencies; these commands do not provision the machine or
replace its kernel. The GPU route requires the [Pi native contract](../backends/pi/docs/CONTRACT.md)
and its built Torch extension. CPU execution does not require that extension.

Choose an absolute work directory outside both repositories. Keep the build,
rendered deployment and receipts there. All paths below are examples to replace.

## CPU build and deployment

After preparing the environment above, install both checkouts from this repository:

```sh
python -m pip install -e '../mstar[asr]' -e .
```

Build the single CPU library using the pinned dependencies and preset in the
[CPU backend guide](../backends/cpu_ggml/README.md):

```sh
python backends/cpu_ggml/build.py --preset pi5 \
  --whisper-source /absolute/whisper.cpp \
  --kleidiai-source /absolute/kleidiai-v1.24.0 \
  --build /absolute/work/build
python tools/whisper/prepare_assets.py \
  --destination /absolute/work/assets --config /absolute/work/cpu.yaml \
  --deployment-template deployments/whisper_tiny_pi_cpu.yaml \
  --cpu-ggml-library /absolute/work/build/libmstar_cpu_encoder.so \
  --cpu-ggml-sha256 HASH_FROM_BUILD_JSON
```

The asset tool checks the pinned multilingual `openai/whisper-tiny` checkpoint
and fixtures, then renders absolute checkpoint and acceleration paths. Add
`--offline` to verify already-present assets without downloading. The runtime
verifies the library hash before loading it. The template places both nodes in
one worker, with FP16 KV storage requested by the decoder, two live requests,
four encoder threads and two decoder threads. It retains the full encoder context.

## Central qualification

Use the finalized centralized harness directly. It infers the pinned model,
fixtures, supported cases and placements from the registry and available hardware;
there is no per-implementation test configuration or numerical-policy selection.
Provide the CPU library as an environment dependency:

```sh
MSTAR_CPU_GGML_LIBRARY=/absolute/work/build/libmstar_cpu_encoder.so \
python -m mstar_accelerators.testing \
  --registry accelerations/whisper/pi_cpu \
  --report /absolute/work/cpu-report.json --require-all-pass
```

Checkpoint lookup uses the standard Hugging Face cache. See
[Qualification](QUALIFICATION.md) for asset identities, acceptance and coverage.
For V3D, select `accelerations/whisper/pi_v3d` and run through
`backends/pi/tools/run_hardware.py --managed`. Never bypass shared admission,
clear quarantine or treat process termination as hardware completion. Serialize
CPU and GPU work on the Pi. Serving does not consume a qualification verdict.

## Ordinary M* service

For CPU, start the ordinary API from a short writable socket directory:

```sh
mkdir -p /absolute/work/uploads /tmp/mstar-pi-demo
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
python -m mstar.api_server.entrypoint \
  --config /absolute/work/cpu.yaml --host 127.0.0.1 --port 8000 \
  --tensor-comm-protocol SHM --socket-path-prefix /tmp/mstar-pi-demo/ \
  --upload-dir /absolute/work/uploads --log-level WARNING
```

This uses the environment's selected M* transport/graph implementation. To
reproduce a measured Rust deployment, use its matching built `mstar_rust` module
and recorded `MSTAR_RUST_GRAPH`, `MSTAR_RUST_ZMQ` and wire-codec settings; do not
attribute its measurements to an unverified or different build.

`tools/whisper/demo.py --config /absolute/work/cpu.yaml --output /absolute/work/service`
starts and stops this same API for service checks. Add `--stochastic-temperature
0.8` for the fourteen-case suite. Accuracy qualification belongs to the centralized harness above.
`--benchmark` selects the retained timing workload, uses WARNING logs and disables
request diagnostics. Service health and timing do not establish model accuracy.

For GPU, render `deployments/whisper_tiny_pi.yaml` with the same asset tool and no
CPU-library arguments. Launch the API or service harness through the managed
launcher, with the matching GPU extension on its Python path. A failed or timed-out
native run must settle through that launcher's ownership protocol before reuse.
