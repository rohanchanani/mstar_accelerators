# Source provenance and retained notices

The native runtime, Torch adapter, managed-provider extension, integration
tests and admission tool were authored for this work. This candidate starts
from accepted Stage 2 commit `0f306a550b5148e5673477675d0aea0542f22189`, including
its canonical DMA-BUF extent repair and the Stage 1 CMake fix
`6607517cc0fd58f9ba705b910d059b7fcac03956`. Historical receipts and obsolete
handoffs are preserved in the delivery evidence and immutable baseline history.

`third_party/ggml/` contains the supplied actual Pi ggml source snapshot from
llama.cpp revision `1b29621bacc6a446bd523fc790316a24a06e886f`, including its r2
storage-binding limit fix, plus the managed-provider extension. The pinned
upstream MIT license is retained as `third_party/ggml/LICENSE`; its original
URL and hash, baseline file manifest, prior patch, and Stage 2 patch are there.
Other embedded source keeps its original license notices. The focused supplied
tree covers the complete include/src/cmake build source used here, not a claim
that every upstream llama.cpp example/test/repository file is vendored.

`tests/fixtures/qpu/source.py` and `programs.json` remain the unchanged observed
r2/Stage 1 programs. `copy_source.py` derives its bounded bit-preserving copy
program from that licensed source and retains the Idein Inc. MIT notice. Its
new machine words have separate source/assembler/hash identities. Their raw-bit
target witness passed in Stage 2. They remain a required regression for the new
provider; that observation does not qualify new complete Whisper programs.

The kernel patch and harness are unchanged from Stage 1. The pinned Raspberry
Pi Linux V3D driver retains its original SPDX identities in the full source.
Its manifest binds the exact original and patched bodies; the extracted host
harness does not replace the driver source or hardware qualification.

The companion M* fork starts from original revision
`fa37d227c3467145a2fe690cd53924ec9d678f90` and retains its own upstream notices.
The provisional Pi factory is not reused. Torch, Vulkan, libdrm, the target kernel, toolchains, and installed
system libraries are external build/runtime dependencies with their own
licenses; no old installed runtime binary is smuggled into the source bundle.
