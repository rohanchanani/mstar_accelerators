#!/usr/bin/env bash
# Install into a dedicated directory; never modify the system ROCm/driver.
set -euo pipefail

usage() {
  echo "Usage: $0 MSTAR_CHECKOUT WORK_DIRECTORY [ROCM_DIRECTORY]" >&2
  echo "Requires Linux x86_64, Python 3.12, uv, git, curl, cargo, c++, and a ROCm 7.2 host." >&2
}
if [[ $# -lt 2 || $# -gt 3 ]]; then usage; exit 2; fi
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  echo "The pinned AMD wheels require Linux x86_64." >&2; exit 2
fi
for cmd in uv git curl cargo c++; do
  command -v "$cmd" >/dev/null || { echo "Missing prerequisite: $cmd" >&2; exit 2; }
done
mstar_checkout=$(realpath "$1")
mkdir -p "$2"
work_directory=$(realpath "$2")
accelerators_checkout=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
rocm_directory=${3:-/opt/rocm-7.2.0}
[[ -f "$mstar_checkout/mstar/model/qwen3_omni/logical.py" && \
   -f "$mstar_checkout/mstar/engine/tensor_forward.py" ]] || {
  echo "Use the paired M* checkout with logical Talker and tensor-forward execution support." >&2; exit 2;
}
[[ -x "$rocm_directory/bin/hipcc" ]] || {
  echo "No hipcc at $rocm_directory/bin/hipcc; pass the ROCm 7.2 toolkit path." >&2; exit 2;
}
export ROCM_PATH="$rocm_directory" ROCM_HOME="$rocm_directory"
export PATH="$rocm_directory/bin:$PATH"
export UV_CACHE_DIR="$work_directory/uv-cache"
constraints="$accelerators_checkout/requirements/qwen3-omni-rocm-constraints.txt"
overrides="$accelerators_checkout/requirements/qwen3-omni-rocm-overrides.txt"
if [[ ! -x "$work_directory/venv/bin/python" ]]; then
  uv venv --python 3.12 --seed "$work_directory/venv"
fi
export VIRTUAL_ENV="$work_directory/venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "Python 3.12 is required")'
uv pip install --python "$VIRTUAL_ENV/bin/python" --torch-backend rocm7.2 \
  -c "$constraints" torch==2.12.1 torchvision==0.27.1 torchaudio==2.11.0
uv pip install --python "$VIRTUAL_ENV/bin/python" --torch-backend rocm7.2 \
  -c "$constraints" --overrides "$overrides" \
  -e "$mstar_checkout[qwen3_omni,audio,dev]" -e "$accelerators_checkout" \
  av soundfile pybind11 ninja cmake pandas einops psutil packaging flydsl 'maturin[patchelf]' aiohttp

mkdir -p "$work_directory/downloads" "$work_directory/vendor"
aiter_wheel=amd_aiter-0.1.25+rocm7.2.manylinux.2.28-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
aiter_path="$work_directory/downloads/$aiter_wheel"
if [[ ! -f "$aiter_path" ]]; then
  curl --fail --location --retry 2 \
    "https://github.com/ROCm/aiter/releases/download/v0.1.25/${aiter_wheel/+/%2B}" \
    --output "$aiter_path.partial"
  mv "$aiter_path.partial" "$aiter_path"
fi
python - "$aiter_path" <<'PY'
import hashlib, pathlib, sys
p = pathlib.Path(sys.argv[1])
with p.open('rb') as source:
    digest = hashlib.file_digest(source, 'sha256').hexdigest()
if digest != '544469800fdfd81a31493f7f635472d9c05ca4b14c69202fa91394076eea1333':
    raise SystemExit(f'AITER wheel SHA256 mismatch: {p}: {digest}')
PY
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps "$aiter_path"

flashinfer_checkout="$work_directory/vendor/amd-flashinfer"
flashinfer_commit=7e2913df7604a647db40ed44dac6bb23575cc838
if [[ ! -d "$flashinfer_checkout" ]]; then
  git init "$flashinfer_checkout"
  git -C "$flashinfer_checkout" fetch --depth 1 https://github.com/ROCm/flashinfer.git "$flashinfer_commit"
  git -C "$flashinfer_checkout" checkout --detach FETCH_HEAD
fi
[[ $(git -C "$flashinfer_checkout" rev-parse HEAD) == "$flashinfer_commit" ]] || {
  echo "Unexpected FlashInfer checkout at $flashinfer_checkout; use a fresh work directory." >&2; exit 2;
}
[[ -z $(git -C "$flashinfer_checkout" status --porcelain) ]] || {
  echo "FlashInfer checkout has local edits; refusing to build it as the pinned release." >&2; exit 2;
}
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AMD_FLASHINFER=0.6.18+amd.1 \
  uv pip install --python "$VIRTUAL_ENV/bin/python" --torch-backend rocm7.2 \
  -c "$constraints" --overrides "$overrides" "$flashinfer_checkout"

# Keep compilation caches separate from other GPUs and software stacks. AITER's
# normal wheel location retains its prebuilt modules; do not hide it with an
# empty AITER_JIT_DIR.
python - "$work_directory" "$rocm_directory" "$mstar_checkout" "$accelerators_checkout" <<'PY'
from pathlib import Path
import shlex, sys
work, rocm, mstar, accelerators = map(Path, sys.argv[1:])
cache = work / 'compile-cache'
variables = {
    'VIRTUAL_ENV': work / 'venv', 'ROCM_PATH': rocm, 'ROCM_HOME': rocm,
    'UV_CACHE_DIR': work / 'uv-cache', 'HF_HOME': work / 'hf',
    'MIOPEN_USER_DB_PATH': cache / 'miopen', 'MIOPEN_CUSTOM_CACHE_DIR': cache / 'miopen',
    'TORCHINDUCTOR_CACHE_DIR': cache / 'inductor', 'TRITON_CACHE_DIR': cache / 'triton',
    'FLASHINFER_WORKSPACE_BASE': cache / 'flashinfer',
    'MSTAR_AMD_WORK_DIRECTORY': work,
    'MSTAR_AMD_MSTAR_CHECKOUT': mstar,
    'MSTAR_AMD_ACCELERATORS_CHECKOUT': accelerators,
}
for name in ('MIOPEN_USER_DB_PATH', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'FLASHINFER_WORKSPACE_BASE'):
    variables[name].mkdir(parents=True, exist_ok=True)
lines = ['# Source this file before preparing assets or launching M*.']
lines += [f'export {key}={shlex.quote(str(value))}' for key, value in variables.items()]
lines += ['export PATH="$VIRTUAL_ENV/bin:$ROCM_PATH/bin:$PATH"']
(work / 'environment.sh').write_text('\n'.join(lines) + '\n')
PY
source "$work_directory/environment.sh"
maturin develop --release --manifest-path "$mstar_checkout/rust/Cargo.toml"
python -c 'import mstar, mstar_rust; print("M*:", mstar.__file__); print("Rust:", mstar_rust.__file__)'
uv pip freeze --python "$VIRTUAL_ENV/bin/python" > "$work_directory/environment-freeze.txt"
echo "Installed. Source $work_directory/environment.sh to run tools/qwen3_omni/run_foundation.py."
