"""Run the generated-file host campaign after the agent fills its source files.

The suite discovers the selected acceleration source directory or file.
It explicitly interprets those programs on CPU; this is not Pi qualification,
Pi emulation, or a CPU fallback in a Pi deployment. Use the retained official
Torch/FFmpeg environment. No native admission or backend initialization occurs.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

IDENTITY = """
import hashlib, json, pathlib, platform, sys, torch
import mstar, mstar_accelerators
from tests.whisper import evaluator, provider
from mstar_accelerators.sources import acceleration_files
source, mstar_source, files = map(pathlib.Path, sys.argv[1:])
assert pathlib.Path(mstar.__file__).resolve() == mstar_source / 'mstar/__init__.py'
assert pathlib.Path(mstar_accelerators.__file__).resolve() == source / 'src/mstar_accelerators/__init__.py'
assert pathlib.Path(evaluator.__file__).resolve() == source / 'tests/whisper/evaluator.py'
assert pathlib.Path(provider.__file__).resolve() == source / 'tests/whisper/provider.py'
identities = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in acceleration_files(files).items()}
assert identities, 'the authoring corpus has no acceleration source files'
print(json.dumps(dict(python=sys.version, executable=sys.executable,
                     platform=platform.platform(), machine=platform.machine(),
                     torch=torch.__version__, torch_path=torch.__file__,
                     cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
                     mstar=mstar.__file__, external=mstar_accelerators.__file__,
                     evaluator=evaluator.__file__, host_provider=provider.__file__,
                     acceleration_sources=str(files), acceleration_sha256=identities,
                     declarations=list(identities)), indent=2))
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "mstar", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--accelerations", type=Path,
                        help="filled source file/directory; defaults to source/accelerations/whisper/pi_v3d")
    args = parser.parse_args(argv)
    source, mstar, checkpoint, output = (getattr(args, key).resolve()
                                       for key in ("source", "mstar", "checkpoint", "output"))
    accelerations = (args.accelerations or source / "accelerations/whisper/pi_v3d").resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ | {
        "PYTHONPATH": os.pathsep.join(map(str, (source / "src", mstar, source))),
        "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "4",
        "MSTAR_WHISPER_CHECKPOINT": str(checkpoint),
        "MSTAR_ACCELERATION_SOURCE": str(accelerations),
        "MSTAR_WHISPER_VARIANTS_ROOT": str(output / "file_variants"),
    }

    def run(name, command, **overrides):
        current = env | overrides
        receipt = {"argv": command, "cwd": str(source), "started_utc": datetime.now(UTC).isoformat(),
                       "status": "starting", "environment": {key: current[key] for key in (
                           "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "HF_HUB_OFFLINE", "OMP_NUM_THREADS",
                           "MSTAR_WHISPER_CHECKPOINT", "MSTAR_WHISPER_AUDIO", "MSTAR_WHISPER_FULL_REPORT",
                           "MSTAR_WHISPER_VARIANTS_ROOT", "MSTAR_ACCELERATION_SOURCE", "LD_LIBRARY_PATH") if key in current}}
        receipt_path = output / f"{name}.command.json"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        before = time.monotonic()
        try:
            with (output / f"{name}.stdout").open("w") as stdout, (output / f"{name}.stderr").open("w") as stderr:
                result = subprocess.run(command, cwd=source, env=current, stdout=stdout, stderr=stderr,
                                        check=False)
            receipt.update(status="exited", exit_code=result.returncode)
        except BaseException as error:
            receipt.update(status="interrupted", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            receipt.update(seconds=time.monotonic() - before, finished_utc=datetime.now(UTC).isoformat())
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"{name}: exit={result.returncode}, seconds={receipt['seconds']:.3f}", flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)

    run("identity", [sys.executable, "-B", "-c", IDENTITY, str(source), str(mstar), str(accelerations)])
    for name, audio, tests in (("full", "jfk16k.wav", "tests"),
                               ("tail", "jfk_tail16k.wav", "tests/whisper/test_full_checkpoint.py")):
        run(name, [sys.executable, "-B", "-m", "pytest", "-q", tests],
            MSTAR_WHISPER_AUDIO=str(source / "tests/fixtures/whisper" / audio),
            MSTAR_WHISPER_FULL_REPORT=str(output / f"{name}.json"))
    run("verify", [sys.executable, "-B", str(Path(__file__).with_name("verify_host_reports.py")),
                   "--source", str(source), "--full", str(output / "full.json"),
                   "--tail", str(output / "tail.json"), "--identity", str(output / "identity.stdout"),
                   "--output", str(output / "verification.json")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
