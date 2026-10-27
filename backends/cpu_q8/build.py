"""Build the optional synchronous CPU Q8 kernel from existing vendored ggml."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ggml', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cc', default='gcc')
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise RuntimeError('this optional build requires Linux AArch64')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('preserve the existing artifact; choose a fresh output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    ggml, wrapper = args.ggml.resolve(strict=True), Path(__file__).with_name('readout.c').resolve()
    flags = ['-O3', '-std=c11', '-fPIC', '-fvisibility=hidden', '-ffunction-sections', '-fdata-sections',
             '-march=armv8.2-a+dotprod', '-I' + str(ggml / 'include'), '-I' + str(ggml / 'src'), '-I' + str(ggml / 'src/ggml-cpu')]
    commands, objects = [], []
    for source, name in ((ggml / 'src/ggml-cpu/arch/arm/quants.c', 'arm-quants'), (wrapper, 'readout')):
        obj = output.parent / (output.stem + '-' + name + '.o')
        objects.append(obj)
        commands.append([args.cc, *flags, '-MMD', '-MF', str(obj.with_suffix('.d')), '-c', str(source), '-o', str(obj)])
    commands.append([args.cc, '-shared', '-Wl,--gc-sections', '-Wl,--no-undefined', *map(str, objects), '-lm', '-o', str(output)])
    report = {'compiler': subprocess.check_output([args.cc, '--version'], text=True).splitlines()[:2], 'commands': commands}
    started = time.perf_counter()
    for index, command in enumerate(commands):
        run = subprocess.run(command, text=True, capture_output=True, timeout=90)
        output.with_suffix(f'.build-{index}.stdout').write_text(run.stdout)
        output.with_suffix(f'.build-{index}.stderr').write_text(run.stderr)
        run.check_returncode()
    report['build_seconds'] = time.perf_counter() - started
    report['library_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    report['library_bytes'] = output.stat().st_size
    report['ldd'] = subprocess.check_output(['ldd', str(output)], text=True)
    report['exports'] = subprocess.check_output(['nm', '-D', '--defined-only', str(output)], text=True)
    disassembly = subprocess.check_output(['objdump', '-d', str(output)], text=True)
    output.with_suffix('.disassembly').write_text(disassembly)
    report['sdot_instruction_sites'] = sum('sdot' in line for line in disassembly.splitlines())
    if not report['sdot_instruction_sites']:
        raise RuntimeError('the linked artifact does not contain the required SDOT implementation')
    report['source_dependencies'] = {}
    for obj in objects:
        for token in shlex.split(obj.with_suffix('.d').read_text().replace('\\\n', ' '))[1:]:
            path = Path(token)
            if path.is_file():
                report['source_dependencies'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    handle = ctypes.CDLL(str(output))
    handle.q8cpu_capability.restype = ctypes.c_int
    if handle.q8cpu_capability() != 1:
        raise RuntimeError('actual runtime HWCAP_ASIMDDP is unavailable')
    report['cpu_q8'] = {'library': str(output), 'sha256': report['library_sha256']}
    output.with_suffix('.build.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
