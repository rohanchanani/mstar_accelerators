# Whisper CPU implementation

This backend implements Whisper encoder and decoder computation using pinned ggml
CPU kernels, KleidiAI and ARM SIMD attention kernels. It does not link Whisper's
model server, load a separate checkpoint or own request state.

## Build

Use ggml from whisper.cpp revision
`d1be6fde11ac6e0407606b4e42fe72d34add8037` and KleidiAI 1.24.0. The source override
is recorded by complete file hashes; supply the retained dependency source rather
than an unreviewed newer release.

```sh
python backends/cpu_ggml/build.py --preset pi5 \
  --whisper-source /absolute/whisper.cpp \
  --kleidiai-source /absolute/kleidiai-v1.24.0 \
  --build /absolute/new-build
```

The preset selects shared attention packing, ARM GEMM, encoder SDOT QK/PV, tiled
softmax, exact guarded PV accumulation, Q8 convolutions, 128-row FFN tiles and the
single-query cross-attention kernel. It requires OpenMP and KleidiAI. Required
numerical guards remain active. The builder validates pinned ggml sources, patches
a private copy, builds from source and records commands, compiler flags, dependency
hashes and the resulting library in `BUILD.json`. `--plan-only` verifies and emits
the build plan without compiling. Nothing modifies the dependency checkout.

## Select and bind

The CPU family contains a Q8 encoder and one combined greedy/stochastic
Q4-KleidiAI decoder:

```yaml
runtime_extension:
  factory: mstar_accelerators.backends.torch.runtime:TorchRuntime
  accelerations: accelerations/whisper/pi_cpu
  threads: 1
  cpu_ggml:
    library: /absolute/new-build/libmstar_cpu_encoder.so
    sha256: <BUILD.json library hash>
  cpu_ggml_threads:
    audio_encoder: 4
    decoder: 2
  cpu_ggml_cross_threads: 4
```

The CPU deployment needs only this verified native library. Model/checkpoint,
placement, concurrency and resource budgets belong in the deployment configuration.

## Computation and ownership

The encoder consumes F32 mel features and returns owned F32 hidden states. Dense
weights use upstream Q8 packing after checkpoint F16 rounding; the selected
convolutions and attention use the preset's physical kernels. All 1500 query
positions are computed. Attention includes the public implementation's padded
1536-key geometry. The centralized numerical policy evaluates this physical
realization against the independent logical reference.

The decoder consumes token IDs, admitted positions and borrowed runtime-owned
FP16 self/cross KV. Its Q8 body and Q4-KleidiAI vocabulary projection share one
prepared physical body across graph walks. Prefill prepares cross KV;
decode reads it. Greedy selection is native; positive-temperature selection uses
the existing CPU categorical implementation and runtime-supplied sampling plan.
Only M* commits offsets and state effects.

Native code checks shapes, extents, positions and aliases before state writes.
Borrowed storage remains protected through synchronous threadpool completion;
no workspace alias is published as an independently owned result. Program slots
guard reusable workspace, and close releases native preparation only after those
slots drain. There is no private KV allocator or second sampling lifecycle.

## Source map

- `encoder.cpp`, `encoder_gelu_lut.h`: encoder graph and fused physical operations.
- `decoder.cpp`, `decoder_{storage,readout,tokens}.h`: decoder graph, borrowed-state
  ABI, vocabulary projection and token entry point.
- `flash_*.h`: attention packing, SDOT, softmax and guarded accumulation kernels.
- `build.py`, `CMakeLists.txt`, `flash_*.py`: pinned source transformations/build.
- `q8_provider.h`: explicit supported upstream weight-buffer provider selection.
- `attention_probe.*` and `tests/`: retained profiling and backend mechanism
  controls, not additional serving implementations or numerical qualification
  entry points. Individual build switches support these controls; `--preset pi5`
  selects the shipped configuration.

Keep third-party licenses with redistributed sources. Accuracy/token/state
qualification belongs to the central consumer, not the builder. Source identity,
compilation, numerical correctness and completed-service performance are separate
claims.
