"""Build the Torch adapter against this source tree's native/ggml libraries.

The root CMake project owns the native runtime and shader compilation. This
package owns only the Torch ABI and Python registration. Build without pip
isolation so the extension uses the exact Torch installed in the worker venv.
"""

from pathlib import Path
import os

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

if torch.__version__.split("+")[0] != "2.10.0":
    raise RuntimeError(f"this source is pinned to Torch 2.10.0, found {torch.__version__}")

build_dir = Path(os.environ.get("MSTAR_PI_BUILD_DIR", ROOT / "build")).resolve()
static_libraries = [build_dir / "lib" / f"lib{name}.a" for name in (
    "mstar_pi_native", "ggml-vulkan", "ggml", "ggml-base",
)]
for library in static_libraries:
    if not library.is_file():
        raise RuntimeError(f"build the root CMake project first; missing {library}")
if not (HERE / "mstar_pi" / "transfer.spv").is_file():
    raise RuntimeError("copy build/shaders/transfer.spv into torch/mstar_pi before packaging")

setup(
    name="mstar-pi",
    version="0.4.0",
    description="Native Pi storage and execution through Torch",
    packages=["mstar_pi"],
    package_data={"mstar_pi": ["transfer.spv"]},
    ext_modules=[CppExtension(
        "mstar_pi._C",
        sources=[str(HERE / "csrc" / name) for name in ("backend.cpp", "ggml_provider.cpp")],
        depends=[str(HERE / "csrc" / "graph_provider.inc")] + [str(p) for p in static_libraries] + [
            str(ROOT / "native" / "include" / "mstar_pi" / name)
            for name in ("runtime.h", "provider.h")
        ],
        include_dirs=[str(ROOT / "native" / "include"), str(ROOT / "third_party" / "ggml" / "include")],
        # ggml's registration/core/provider archives have cross references. The
        # group is link-time resolution, not dynamic provider discovery.
        extra_link_args=[
            "-Wl,--start-group", *map(str, static_libraries), "-Wl,--end-group",
            # Keep these after the archives which introduce their references;
            # --as-needed must not discard them before scanning the group.
            "-lvulkan", "-ldrm",
        ],
        extra_compile_args=["-std=c++20", "-O2", "-g0", "-Wall", "-Wextra"],
    )],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.11",
    install_requires=["torch==2.10.0"],
)
