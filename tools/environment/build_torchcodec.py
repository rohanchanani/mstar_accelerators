#!/usr/bin/env python3
"""Build pinned TorchCodec against the current Torch and task-local FFmpeg.

No system installation or upgrade is performed. APT resolves and downloads
FFmpeg runtime/development packages and their missing dependency closure into
an isolated state; the packages are extracted locally. Each attempt keeps
its own source extraction, build directory, logs, wheel and identity receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import pwd
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import time
import urllib.request


COMMIT = "0b261b98080925f2b709712a5491a1e8dd817065"
ARCHIVE_URL = f"https://codeload.github.com/meta-pytorch/torchcodec/tar.gz/{COMMIT}"
ARCHIVE_SHA256 = "a3a6adf464221269a7281a77cf85b37081e5602d7dae526e985849c2487c5831"
COMPONENTS = ("avdevice", "avfilter", "avformat", "avcodec", "avutil", "swresample", "swscale")


def ffmpeg_package(name):
    return name == "ffmpeg" or any(name.startswith("lib" + part) for part in (*COMPONENTS, "postproc"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Build:
    def __init__(self, work):
        self.work = work.resolve()
        self.run = self.work / "attempts" / str(time.time_ns())
        self.run.mkdir(parents=True)
        self.env = dict(os.environ)
        self.step = 0
        self.receipt = {"status": "RUNNING", "attempt": str(self.run), "commands": []}

    def save(self):
        (self.run / "receipt.json").write_text(json.dumps(self.receipt, indent=2, sort_keys=True) + "\n")

    def command(self, args, *, cwd=None, env=None):
        self.step += 1
        log = self.run / f"{self.step:02d}-{Path(str(args[0])).name}.log"
        record = {"argv": [str(x) for x in args], "cwd": str(cwd or self.run), "log": str(log)}
        self.receipt["commands"].append(record)
        self.save()
        print(f"[{self.step}] {' '.join(record['argv'])}\n  log: {log}", flush=True)
        with log.open("w") as output:
            result = subprocess.run(record["argv"], cwd=cwd or self.run,
                                    env=env or self.env, stdout=output, stderr=subprocess.STDOUT)
        record["exit_code"] = result.returncode
        self.save()
        if result.returncode:
            raise RuntimeError(f"command failed ({result.returncode}); preserved log: {log}")
        return log.read_text()

    def configure_apt(self, installed):
        apt = self.run / "apt"
        for subdir in ("state/lists/partial", "cache/archives/partial", "log", "empty", "packages"):
            (apt / subdir).mkdir(parents=True)
        # APT sees the real installed dependency base, except for FFmpeg:
        # obtain its complete matching runtime/header set in our own prefix
        # even if the host happens to have a copy already. APT owns dependency
        # resolution. Its install operation below is strictly download-only.
        status = []
        for stanza in Path("/var/lib/dpkg/status").read_text().split("\n\n"):
            match = re.search(r"^Package: (.+)$", stanza, re.MULTILINE)
            if match and not ffmpeg_package(match[1]):
                status.append(stanza)
        (apt / "state/status").write_text("\n\n".join(status) + "\n")
        preferences = "\n".join(
            f"Package: {name}\nPin: version {version}\nPin-Priority: 1001\n"
            for name, (version, _arch) in installed.items() if not ffmpeg_package(name)
        )
        (apt / "preferences").write_text(preferences)
        config = apt / "apt.conf"
        # APT_CONFIG is read before the ordinary configuration fragments. This
        # keeps all lists/caches/logs and update hooks local, while retaining
        # the machine's configured sources and signature keyrings.
        config.write_text(
            'Dir::Etc::main "/dev/null";\n'
            f'Dir::Etc::parts "{apt / "empty"}";\n'
            'Dir::Etc::sourcelist "/etc/apt/sources.list";\n'
            'Dir::Etc::sourceparts "/etc/apt/sources.list.d";\n'
            f'Dir::State "{apt / "state"}";\n'
            f'Dir::State::status "{apt / "state/status"}";\n'
            f'Dir::Etc::preferences "{apt / "preferences"}";\n'
            f'Dir::Etc::preferencesparts "{apt / "empty"}";\n'
            f'Dir::Cache "{apt / "cache"}";\n'
            f'Dir::Log "{apt / "log"}";\n'
            f'APT::Sandbox::User "{pwd.getpwuid(os.getuid()).pw_name}";\n'
            'Acquire::Languages "none";\n'
        )
        apt_env = {**self.env, "APT_CONFIG": str(config), "LC_ALL": "C"}
        self.command(["/usr/bin/apt-get", "-o", "APT::Update::Error-Mode=any", "update"], env=apt_env)
        self.receipt["apt_sources"] = {
            str(path): {"sha256": digest(path), "contents": path.read_text()}
            for path in [Path("/etc/apt/sources.list"), *Path("/etc/apt/sources.list.d").glob("*")]
            if path.is_file() and (path.suffix in (".list", ".sources") or path.name == "sources.list")
        }
        return apt, apt_env

    def prepare_ffmpeg(self, prefix=None, requested_version=None):
        multiarch = sysconfig.get_config_var("MULTIARCH")
        raw = self.command(["dpkg-query", "-W", "-f=${Package}\t${Version}\t${Architecture}\t${db:Status-Status}\n"])
        packages = {}
        for line in raw.splitlines():
            name, version, arch, state = line.split("\t")
            if state == "installed":
                packages[name] = (version, arch)
        self.receipt["installed_ffmpeg"] = {name: {"version": v, "architecture": arch}
                                             for name, (v, arch) in packages.items() if ffmpeg_package(name)}

        if prefix is None:
            apt, apt_env = self.configure_apt(packages)
            prefix = self.run / "ffmpeg-prefix"
            prefix.mkdir()
            available = self.command(["/usr/bin/apt-cache", "madison", "libavcodec-dev"], env=apt_env)
            versions = [line.split("|")[1].strip() for line in available.splitlines() if "|" in line]
            installed = [v for name, (v, _arch) in packages.items()
                         if re.fullmatch(r"libavcodec(?:-extra)?(?:58|59|60|61|62)", name)]
            version = requested_version or next((v for v in installed if v in versions), None)
            if version is None:
                policy = self.command(["/usr/bin/apt-cache", "policy", "libavcodec-dev"], env=apt_env)
                version = re.search(r"Candidate:\s*(\S+)", policy)[1]
            info = self.command(["/usr/bin/apt-cache", "show", f"libavcodec-dev={version}"], env=apt_env)
            codec_major = int(re.search(r"libavcodec(\d+)\s*\(=", info)[1])
            if not 58 <= codec_major <= 62:
                raise RuntimeError(f"Configured repositories selected FFmpeg outside TorchCodec 0.10's 4–8 range: {version}")
            self.receipt.update(ffmpeg_major=codec_major - 54, ffmpeg_package_version=version)
            requested = [f"lib{x}-dev={version}" for x in COMPONENTS]
            if not shutil.which("pkg-config"):
                requested.append("pkgconf-bin")
            plan = self.command(["/usr/bin/apt-get", "--simulate", "--no-install-recommends", "--no-remove",
                                 "install", *requested], env=apt_env)
            # The isolated multimedia prefix must not substitute a newer copy
            # of the established Vulkan, Mesa, C/C++ runtime or other installed
            # dependency. Installed-version pins normally enforce this; retain
            # an explicit check of APT's actual proposed plan before download.
            for name, new_version in re.findall(r"^Inst (\S+)(?: \[[^]]+\])? \((\S+)", plan, re.MULTILINE):
                name = name.split(":")[0]
                if name in packages and not ffmpeg_package(name) and packages[name][0] != new_version:
                    raise RuntimeError(f"APT proposes replacing existing dependency {name}: {packages[name][0]} -> {new_version}")
            self.command(["/usr/bin/apt-get", "--assume-yes", "--download-only", "--no-install-recommends",
                          "--no-remove", "install", *requested], env=apt_env)
            debs = sorted((apt / "cache/archives").glob("*.deb"))
            self.receipt["downloaded_packages"] = [{"file": str(p), "sha256": digest(p)} for p in debs]
            for package in debs:
                self.command(["dpkg-deb", "-x", package, prefix])
        prefix = prefix.resolve()
        libdir = prefix / "usr/lib" / multiarch
        # A complete prefix resolves its linker links locally. A previously
        # prepared headers-only prefix may reuse the exact installed runtime;
        # retain those concrete targets and hashes in either case.
        links = {}
        for link in sorted(libdir.glob("lib*.so")):
            if not link.is_symlink():
                continue
            if link.exists():
                target = link.resolve()
                links[str(link)] = {"target": str(target), "sha256": digest(target)}
                continue
            name = os.readlink(link)
            targets = [Path("/usr/lib") / multiarch / name, Path("/lib") / multiarch / name]
            target = next((p.resolve() for p in targets if p.exists()), None)
            if target is None:
                raise RuntimeError(f"No installed runtime resolves development link {link} -> {name}")
            link.unlink()
            link.symlink_to(target)
            links[str(link)] = {"target": str(target), "sha256": digest(target)}
        self.receipt["runtime_links"] = links
        # Extraction does not run Debian's update-alternatives scripts. Some
        # dependencies (notably BLAS/LAPACK) keep their SONAMEs in subdirectories
        # which the loader does not search. Expose their actual ELF SONAMEs in
        # one task-local directory; conflicting providers must not be guessed.
        sonames = prefix / "runtime-sonames"
        sonames.mkdir(exist_ok=True)
        resolved = {}
        libraries = {p.resolve() for p in libdir.glob("*/lib*.so*") if p.is_file()}
        for library in sorted(libraries):
            if library.open("rb").read(4) != b"\x7fELF":
                continue
            dynamic = self.command(["readelf", "-d", library])
            match = re.search(r"\(SONAME\).*\[([^]]+)\]", dynamic)
            if match is None or (libdir / match[1]).exists():
                continue
            name = match[1]
            if Path(name).name != name:
                raise RuntimeError(f"Invalid dependency SONAME: {name}")
            previous = resolved.get(name)
            if previous and previous["sha256"] != digest(library):
                raise RuntimeError(f"Conflicting task-local dependency SONAME: {name}")
            link = sonames / name
            if link.is_symlink():
                if link.resolve() != library:
                    raise RuntimeError(f"Conflicting existing dependency link: {link}")
            elif link.exists():
                raise RuntimeError(f"Dependency link path is already occupied: {link}")
            else:
                link.symlink_to(library)
            resolved[name] = {"target": str(library), "sha256": digest(library)}
        self.receipt["runtime_sonames"] = resolved
        pkg_config = shutil.which("pkg-config") or str(prefix / "usr/bin/pkgconf")
        self.env.update(
            PKG_CONFIG=pkg_config,
            PKG_CONFIG_PATH=str(libdir / "pkgconfig"),
            PKG_CONFIG_SYSROOT_DIR=str(prefix),
            LD_LIBRARY_PATH=os.pathsep.join((str(libdir), str(prefix / "lib" / multiarch),
                                             str(prefix / "usr/lib"), str(prefix / "lib"), str(sonames)))
                            + (":" + self.env["LD_LIBRARY_PATH"] if self.env.get("LD_LIBRARY_PATH") else ""),
        )
        self.command([pkg_config, "--modversion", *("lib" + x for x in COMPONENTS)])
        self.command([pkg_config, "--cflags", "--libs", *("lib" + x for x in COMPONENTS)])
        self.receipt["ffmpeg_prefix"] = str(prefix)
        self.runtime_library_path = os.pathsep.join((str(libdir), str(prefix / "lib" / multiarch),
                                                    str(prefix / "usr/lib"), str(prefix / "lib"), str(sonames)))

    def build(self, args):
        import torch
        if torch.__version__.split("+")[0] != "2.10.0":
            raise RuntimeError(f"This environment recipe is for Torch 2.10.0, found {torch.__version__}")
        if not (3, 10) <= sys.version_info[:2] <= (3, 14):
            raise RuntimeError("TorchCodec 0.10 supports Python 3.10–3.14")
        self.receipt["environment"] = {"python": sys.version, "executable": sys.executable,
                                       "machine": platform.machine(), "torch": torch.__version__,
                                       "torch_cxx11_abi": torch.compiled_with_cxx11_abi()}
        self.prepare_ffmpeg(args.ffmpeg_prefix, args.ffmpeg_version)
        requirements = Path(__file__).with_name("torchcodec-build.txt")
        self.command([sys.executable, "-m", "pip", "install", "--no-deps", "-r", requirements])
        import pybind11
        # sys.base_prefix also handles relocatable host Python distributions
        # whose compiled-in LIBDIR still names their original install prefix.
        self.env["CMAKE_PREFIX_PATH"] = os.pathsep.join((pybind11.get_cmake_dir(), sys.base_prefix))
        self.env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + self.env.get("PATH", "")
        archive = self.work / "torchcodec-0.10.0.tar.gz"
        if not archive.exists():
            with urllib.request.urlopen(ARCHIVE_URL, timeout=120) as response:
                archive.write_bytes(response.read())
        if digest(archive) != ARCHIVE_SHA256:
            raise ValueError(f"TorchCodec source identity mismatch: {archive}")
        self.receipt["source"] = {"commit": COMMIT, "url": ARCHIVE_URL, "sha256": ARCHIVE_SHA256}
        with tarfile.open(archive) as source:
            source.extractall(self.run / "source", filter="data")
        tree = self.run / "source" / f"torchcodec-{COMMIT}"
        self.env.pop("BUILD_AGAINST_ALL_FFMPEG_FROM_S3", None)
        self.env.update(
            BUILD_VERSION="0.10.0", ENABLE_CUDA="OFF", CMAKE_BUILD_TYPE="Release",
            CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs),
            TORCHCODEC_CMAKE_BUILD_DIR=str(self.run / "build"),
            TORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR="OFF",
            # Upstream's explicit local-FFmpeg wheel path. This wheel is for
            # the current environment; it does not bundle FFmpeg binaries.
            I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION="1",
        )
        # Seed the ordinary CMake cache with this interpreter's development
        # artifacts. This avoids selecting a different installed Python and
        # handles an interpreter relocated after its LIBDIR was compiled in.
        library_dirs = (Path(sysconfig.get_config_var("LIBDIR")),
                        Path(sys.base_prefix) / "lib",
                        Path(sys.base_prefix) / "lib" / sysconfig.get_config_var("MULTIARCH"))
        library_names = (sysconfig.get_config_var("LDLIBRARY"), sysconfig.get_config_var("INSTSONAME"))
        python_library = next((directory / name for directory in library_dirs for name in library_names
                               if name and (directory / name).is_file()), None)
        if python_library is None:
            raise RuntimeError("This Python requires its matching shared development library; no libpython was found")
        include = Path(sysconfig.get_path("include"))
        self.command(["cmake", "-S", tree, "-B", self.run / "build",
                      f"-DTorch_DIR={Path(torch.utils.cmake_prefix_path) / 'Torch'}",
                      f"-DPYTHON_VERSION={sys.version_info.major}.{sys.version_info.minor}",
                      f"-DPython_EXECUTABLE={sys.executable}", f"-DPython3_EXECUTABLE={sys.executable}",
                      f"-DPython_LIBRARY={python_library}", f"-DPython3_LIBRARY={python_library}",
                      f"-DPython_INCLUDE_DIR={include}", f"-DPython3_INCLUDE_DIR={include}",
                      "-DENABLE_CUDA=OFF", "-DCMAKE_BUILD_TYPE=Release"])
        wheels = self.run / "wheels"
        self.command([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                      "--wheel-dir", wheels, tree])
        [wheel] = list(wheels.glob("torchcodec-*.whl"))
        self.receipt["wheel"] = {"path": str(wheel), "sha256": digest(wheel)}
        if args.install:
            self.command([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", wheel])
            self.command([sys.executable, Path(__file__).with_name("verify_torchcodec.py"),
                          "--audio", args.audio, "--spawn"])
        environment = self.work / "environment.sh"
        environment.write_text(
            "# Generated for this verified task-local TorchCodec/FFmpeg environment.\n"
            f"export LD_LIBRARY_PATH={shlex.quote(self.runtime_library_path)}"
            "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n"
        )
        self.receipt["runtime_environment"] = str(environment)
        self.receipt["status"] = "PASS"
        self.save()
        print(json.dumps(self.receipt, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--ffmpeg-prefix", type=Path, help="reuse an already extracted development prefix")
    parser.add_argument("--ffmpeg-version", help="select an exact compatible FFmpeg package version in configured sources")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--audio", type=Path, default=Path(__file__).resolve().parents[2] / "tests/fixtures/whisper/jfk16k.wav")
    args = parser.parse_args()
    build = Build(args.work)
    try:
        build.build(args)
    except BaseException as error:
        build.receipt.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        build.save()
        raise


if __name__ == "__main__":
    main()
