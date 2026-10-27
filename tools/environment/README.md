# TorchCodec in the Whisper Pi environment

M*'s retained `Model.load_audio` uses `torchcodec.decoders.AudioDecoder` to
produce mono 16 kHz CPU samples. Keep that implementation. The Pi's model
computation remains in its native ggml programs; this dependency supplies the
ordinary audio-file input path.

## Why this is a source build

The upstream M* dependency floor requires Torch 2.10 or newer. This Pi recipe
pairs Torch 2.10.0 with TorchCodec 0.10.0, following the upstream compatibility
table. The frozen Torch 2.9.1 / TorchCodec 0.8.1 environment remains a separate
behavior control; do not upgrade it in place. The candidate environment and
native Torch adapter must be rebuilt and qualified before claiming parity.

The pinned CPU source build uses the installed Torch package and dynamically
links FFmpeg 4–8 discovered through pkg-config. No system package replacement or
fallback decoder is required by this recipe.

Pinned source:

- Version: `0.10.0`.
- Official tag commit: `0b261b98080925f2b709712a5491a1e8dd817065`.
- Archive: <https://codeload.github.com/meta-pytorch/torchcodec/tar.gz/0b261b98080925f2b709712a5491a1e8dd817065>.
- Archive SHA256: `a3a6adf464221269a7281a77cf85b37081e5602d7dae526e985849c2487c5831`.
- Source compatibility and build instructions: `README.md`, `CONTRIBUTING.md`,
  `setup.py`, and `src/torchcodec/_core/CMakeLists.txt` in that archive.
- Wheel indexes: <https://download.pytorch.org/whl/cpu/torchcodec/> and
  <https://download.pytorch.org/whl/torchcodec/>.

The release-only `BUILD_AGAINST_ALL_FFMPEG_FROM_S3` path downloads x86 Linux
FFmpeg binaries. It is deliberately unset. The source build selects the
ordinary local-FFmpeg CPU path, without changing upstream source.

## Run on the Pi before model/API gates

Use the fresh candidate environment containing official AArch64 Torch 2.10.0
and the candidate's Python requirements. The compiler, Python development
headers, APT tools and Debian/Pi signature keyrings come from the established
machine environment. This command performs no privileged operation and does
not initialize the Pi GPU:

```bash
python tools/environment/build_torchcodec.py \
  --work "$MSTAR_DEMO_WORK/torchcodec" \
  --jobs 1 \
  --install
source "$MSTAR_DEMO_WORK/torchcodec/environment.sh"
```

`MSTAR_DEMO_WORK` is the same task-specific work directory used by the recipe,
outside both source repositories. One compiler job bounds the build's memory
use; it is unrelated to inference parallelism.

The tool:

1. Records Python, Torch, C++ ABI and installed FFmpeg runtime packages.
2. Creates isolated APT lists/cache/log directories and a private installed
   package status, then verifies the machine's configured repositories with
   their ordinary signature policy. APT resolves a complete matching FFmpeg
   runtime/header set, including missing runtime dependencies and pkg-config
   when needed. It prefers the installed compatible FFmpeg version if that
   exact version remains available; otherwise it selects the configured
   repository candidate, constrained to FFmpeg 4–8. `--ffmpeg-version` can
   select an exact compatible version explicitly. Every requested FFmpeg
   component is pinned to that one version for the solve and download.
3. Executes APT's **download-only** operation against that private state and
   extracts the resulting `.deb` files into the attempt's prefix. All other
   installed packages are pinned at their existing versions; the actual plan
   is checked before downloading. This cannot silently substitute newer Mesa,
   Vulkan, libc or other established runtime dependencies. There is no system
   package installation, root helper call or system upgrade. The signed
   indexes, solver plan, downloaded packages, hashes and linker targets remain
   with the receipt.
4. Installs the pinned Python build tools from `torchcodec-build.txt`, verifies
   the immutable source archive, and extracts a fresh source tree per attempt.
   It supplies this interpreter's actual headers/library to CMake, disables
   CUDA, and builds against the current Torch. Warnings-as-errors remain on.
5. Produces a local wheel and its SHA256. The upstream explicit local-FFmpeg
   wheel option is used; the resulting wheel contains TorchCodec libraries,
   not a bundled FFmpeg distribution. It is an artifact for this environment.
6. With `--install`, installs that wheel without resolving/replacing Torch and
   runs the real `AudioDecoder` call in both the parent and a spawned process.
   The fixed PCM16 fixture must match an independent PCM decoding exactly;
   loaded TorchCodec/FFmpeg library paths are included in the result.
7. Writes `environment.sh` with the task-local library search path. Source
   this file before the M* API demo and any subsequent decoder checks. Spawned
   workers inherit this environment. Retain the referenced prefix for the
   life of this deployment environment.

Every attempt has its own `receipt.json`, command logs, source, CMake build
directory and wheel. A failed attempt is retained. A retry creates a new
attempt; it does not overwrite the failed logs or alter the oracle.

Preinstalled FFmpeg is not required. The seven runtime components are
`libavdevice`, `libavfilter`, `libavformat`, `libavcodec`, `libavutil`,
`libswresample`, and `libswscale`; APT also resolves their declared dependency
closure. Already installed non-FFmpeg dependencies are reused. The earlier Pi
packets did not inventory these components, so target receipts must establish
what was used. If the configured repositories cannot supply a compatible
FFmpeg set without replacing an installed base dependency, the solver failure
is a concrete environment conflict. Preserve it; do not bypass the pins or
silently upgrade Torch, Python or the OS. A previously verified task-local
FFmpeg prefix can be reused with `--ffmpeg-prefix PATH`.

To repeat the installed-library check without rebuilding:

```bash
source "$MSTAR_DEMO_WORK/torchcodec/environment.sh"
python tools/environment/verify_torchcodec.py \
  --audio tests/fixtures/whisper/jfk16k.wav \
  --spawn
```

The expected records are `PASS` for `parent` and `spawn`, Torch `2.10.0`,
TorchCodec `0.10.0`, 16 kHz, one channel and identical PCM hashes. Actual target
build/import/decoding receipts are required before the complete M* API demo.
This check establishes audio input handling, not Pi GPU execution or speed.
