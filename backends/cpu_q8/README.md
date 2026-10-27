# Optional ARM CPU Q8 readout

This synchronous C ABI calls the existing vendored ggml ARM block-Q8 quantizer
and SDOT vector dot. It borrows its input, immutable prepared weights and output
pointers for one call. Its activation scratch is on the stack (408 bytes at
K=384). It owns no tensor storage, thread pool, cache or serving lifecycle.

Build on Linux AArch64 with an existing GCC and dot-product capable CPU:

```sh
python backends/cpu_q8/build.py \
  --ggml backends/pi/third_party/ggml --output /absolute/new/path/libq8readout.so
```

Only the ARM quants translation unit and the small wrapper compile. The build
retains commands, dependency hashes, compiler version, library hash, exports,
disassembly and dynamic dependencies beside the artifact. Serving never builds
or downloads a library. Reuse an independently measured artifact only with its
exact hash and retained build receipt.

The Pi CPU deployment retains this artifact's explicit ARM capability check.
Configure `runtime_extension.cpu_q8` with its absolute library path and SHA256.
The worker verifies the bytes and hardware capability; missing capability is not
an implicit fallback. The shipped decoder's Q4 vocabulary computation is in
`backends/cpu_ggml`, not in a second model-serving implementation here.

The build and library remain separate from model state and sampling ownership.
Use `deployments/whisper_tiny_pi_cpu.yaml` and the main CPU backend guide for the
complete deployment, including the ggml artifact and runtime-owned FP16 cache.
