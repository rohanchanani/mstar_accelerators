"""Build an optional CPU-only ggml encoder; never build or link Whisper itself."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

PIN = 'd1be6fde11ac6e0407606b4e42fe72d34add8037'
# Physical choices of the retained Pi 5 implementation; not model semantics.
PI5_FLAGS = (
    'shared_flash_pack', 'arm_simd_gemm', 'encoder_qk_sdot',
    'encoder_softmax_tile', 'encoder_softmax_xnn', 'encoder_pv_sdot',
    'encoder_pv_gemm4x8', 'encoder_pquant_vector_store',
    'encoder_pv_exact_accumulate', 'decoder_cross_q1',
)

ARCHIVE_INVENTORY = 'abbf71ed47e660ddb1a81b854d28eb7bbbd16d1fc337da0817a068c934627763'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(source, inventory):
    if inventory is not None:
        if sha(inventory) != ARCHIVE_INVENTORY:
            raise ValueError('ggml archive inventory SHA256 differs from the pinned source')
        value = json.loads(inventory.read_text())
        if value['whisper_revision'] != PIN:
            raise ValueError('ggml source revision differs')
        root = source / 'ggml'
        actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
        if actual != set(value['files']):
            raise ValueError('ggml source file inventory differs')
        for name, expected in value['files'].items():
            path = root / name
            if path.is_symlink() or path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
                raise ValueError('ggml source bytes differ: ' + name)
        return PIN, value['ggml_tree'], sha(inventory)
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != PIN:
        raise ValueError('encoder build requires the pinned upstream Whisper/ggml source')
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', 'HEAD', '--', 'ggml'], check=True)
    if subprocess.check_output(['git', '-C', str(source), 'ls-files', '--others', '--exclude-standard', 'ggml'], text=True).strip():
        raise ValueError('untracked files in the pinned ggml source closure')
    tree = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD:ggml'], text=True).strip()
    return revision, tree, None


def physical_source(source, build, shared_pack, arm_simd, qk_sdot=False, softmax_tile=False, pv_sdot=False, pv_gemm=False, softmax_xnn=False, pquant_vector_store=False, decoder_cross_q1=False, pv_exact_accumulate=False):
    """Apply explicit physical overlays to a private, verified source copy."""
    root = source / 'ggml'
    if pv_exact_accumulate and not (pv_sdot and pv_gemm and pquant_vector_store):
        raise ValueError('Exact PV accumulation requires PV SDOT, 4x8 GEMM and vector P stores')
    if pquant_vector_store and not pv_sdot:
        raise ValueError('P vector stores require explicit encoder PV SDOT')
    if softmax_xnn and not softmax_tile:
        raise ValueError('encoder XNN exp requires tile softmax')
    if pv_gemm and not pv_sdot:
        raise ValueError('PV GEMM 4x8 requires explicit PV SDOT')
    if pv_sdot and not (qk_sdot and softmax_tile):
        raise ValueError("PV SDOT requires explicit QK SDOT and tile softmax")
    if softmax_tile and not qk_sdot:
        raise ValueError('encoder tile softmax requires explicit QK SDOT')
    if qk_sdot and not (shared_pack and arm_simd):
        raise ValueError('QK SDOT requires shared packing and ARM SIMD overlays')
    if not (shared_pack or arm_simd or decoder_cross_q1):
        return root, {}
    patched = build / 'ggml-source'
    shutil.copytree(root, patched)
    cpu = patched / 'src/ggml-cpu'
    changes = {}

    def write(name, text):
        path = cpu / name
        key = 'src/ggml-cpu/' + name
        original = changes[key]['original_sha256'] if key in changes else (sha(path) if path.exists() else None)
        changes[key] = {'original_sha256': original,
                        'patched_sha256': hashlib.sha256(text.encode()).hexdigest()}
        path.write_text(text)

    if shared_pack:
        from flash_shared_pack import ops_overlay, planner_overlay
        write('ops.cpp', ops_overlay((cpu / 'ops.cpp').read_text()))
        write('ggml-cpu.c', planner_overlay((cpu / 'ggml-cpu.c').read_text()))
        write('flash_shared_pack.h', Path(__file__).with_name('flash_shared_pack.h').read_text())
    if qk_sdot:
        if not shared_pack or not arm_simd:
            raise ValueError('QK SDOT requires shared packing and ARM SIMD overlays')
        from flash_qk_sdot import ops_overlay as qk_ops_overlay
        write('ops.cpp', qk_ops_overlay((cpu / 'ops.cpp').read_text()))
        write('flash_qk_sdot.h', Path(__file__).with_name('flash_qk_sdot.h').read_text())
    if softmax_tile:
        from flash_softmax_tile import ops_overlay as softmax_ops_overlay
        write('ops.cpp', softmax_ops_overlay((cpu / 'ops.cpp').read_text()))
        write('flash_softmax_tile.h', Path(__file__).with_name('flash_softmax_tile.h').read_text())
    if pv_sdot:
        from flash_pv_sdot import ops_overlay as pv_ops_overlay
        write("ops.cpp", pv_ops_overlay((cpu / "ops.cpp").read_text()))
        write("flash_pv_sdot.h", Path(__file__).with_name("flash_pv_sdot.h").read_text())
    if pv_gemm:
        from flash_pv_gemm import ops_overlay as pv_gemm_overlay
        write('ops.cpp', pv_gemm_overlay((cpu / 'ops.cpp').read_text()))
        write('flash_pv_gemm.h', Path(__file__).with_name('flash_pv_gemm.h').read_text())
    if pquant_vector_store:
        from flash_pquant_vector_store import ops_overlay as pquant_overlay
        write('ops.cpp', pquant_overlay((cpu / 'ops.cpp').read_text()))
        write('flash_pquant_vector_store.h', Path(__file__).with_name('flash_pquant_vector_store.h').read_text())
    if pv_exact_accumulate:
        from flash_pv_exact_accumulate import ops_overlay as pv_exact_overlay
        write('ops.cpp', pv_exact_overlay((cpu / 'ops.cpp').read_text()))
        write('flash_pv_exact_accumulate.h', Path(__file__).with_name('flash_pv_exact_accumulate.h').read_text())
    if softmax_xnn:
        write('flash_softmax_xnn.h', Path(__file__).with_name('flash_softmax_xnn.h').read_text())
        write('XNNPACK_EXP_LICENSE', Path(__file__).with_name('XNNPACK_EXP_LICENSE').read_text())
        definition='    target_compile_definitions(${GGML_CPU_NAME} PRIVATE MSTAR_ENCODER_SOFTMAX_XNN=1)\n'
        original='    target_sources(${GGML_CPU_NAME} PRIVATE ${GGML_CPU_SOURCES})'
        cmake=(cpu / 'CMakeLists.txt').read_text()
        if cmake.count(original)!=1:
            raise ValueError('expected unique CPU target source declaration')
        write('CMakeLists.txt', cmake.replace(original,definition+original))
    if decoder_cross_q1:
        from flash_cross_q1 import ops_overlay as cross_q1_overlay
        write('ops.cpp', cross_q1_overlay((cpu / 'ops.cpp').read_text()))
        for name in ('flash_cross_q1.h', 'flash_cross_exp_horner.h', 'XNNPACK_EXP_LICENSE'):
            write(name, Path(__file__).with_name(name).read_text())
    if arm_simd:
        from flash_arm_simd import HEADER_SHA256, header_overlay
        if sha(cpu / 'simd-gemm.h') != HEADER_SHA256:
            raise ValueError('SIMD GEMM header differs from pinned upstream')
        write('simd-gemm.h', header_overlay((cpu / 'simd-gemm.h').read_text()))
    return patched, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=('pi5',), help='retained Pi 5 CPU implementation')
    parser.add_argument('--whisper-source', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--openmp', choices=('ON', 'OFF'), default='ON')
    parser.add_argument('--source-manifest', type=Path, help='pinned complete ggml inventory for a Git-free source archive')
    parser.add_argument('--shared-flash-pack', action='store_true', help='existing operator-owned once-per-call F16 K/V packing')
    parser.add_argument('--arm-simd-gemm', action='store_true', help='admit the standard ARM NEON macro for upstream GEMM blocking')
    parser.add_argument('--encoder-qk-sdot', action='store_true', help='explicit encoder-only transient Q8 QK experiment')
    parser.add_argument('--encoder-softmax-tile', action='store_true', help='encoder-only whole-tile ARM softmax using pinned exp primitive')
    parser.add_argument('--encoder-softmax-xnn', action='store_true', help='encoder-tagged guarded pinned XNN normal exp core; default unchanged')
    parser.add_argument('--encoder-pv-sdot', action='store_true', help='encoder-only transient Q8 probability/value multiplication')
    parser.add_argument('--encoder-pv-gemm4x8', action='store_true', help='encoder-only 4x8 PV SDOT tile with canonical packing')
    parser.add_argument('--encoder-pquant-vector-store', action='store_true', help='exact encoder P-only non-saturating packed vector stores; default unchanged')
    parser.add_argument('--encoder-pv-exact-accumulate', action='store_true', help='encoder-only guarded PV update retaining original products/sums/final addition; default unchanged')
    parser.add_argument('--decoder-cross-q1', action='store_true', help='guarded Q1 D64 F16 cross FLASH with independently pinned Horner exp; default unchanged')
    parser.add_argument('--encoder-q8-conv', type=int, choices=(0, 1, 2, 3), default=0,
                        help='physical convolution mask: 0=F16 default, 1=conv1, 2=conv2, 3=both')
    parser.add_argument('--encoder-ffn-tile', type=int, choices=(0, 128), default=0,
                        help='physical complete FFN query tile: 0=full-query default, 128=local complete blocks')
    parser.add_argument('--kleidiai-source', type=Path, help='explicit retained KleidiAI source; use without downloading a second dependency')
    parser.add_argument('--plan-only', action='store_true', help='verify inputs, prepare private source and record commands without compiling')
    args = parser.parse_args()
    if args.preset == 'pi5':
        if args.kleidiai_source is None:
            parser.error('--preset pi5 requires --kleidiai-source')
        if args.openmp != 'ON':
            parser.error('--preset pi5 requires OpenMP')
        if args.encoder_q8_conv not in (0, 3) or args.encoder_ffn_tile not in (0, 128):
            parser.error('physical options conflict with --preset pi5')
        for name in PI5_FLAGS:
            setattr(args, name, True)
        args.encoder_q8_conv = 3
        args.encoder_ffn_tile = 128
    source = args.whisper_source.resolve(strict=True)
    build = args.build.resolve()
    if build.exists():
        raise FileExistsError('choose a fresh build directory; preserve earlier artifacts')
    revision, tree, inventory_sha = source_identity(source, args.source_manifest)
    if not 1 <= args.jobs <= 8:
        raise ValueError('invalid build job count')
    here = Path(__file__).resolve().parent
    build.mkdir(parents=True)
    ggml_source, patches = physical_source(source, build, args.shared_flash_pack, args.arm_simd_gemm, args.encoder_qk_sdot, args.encoder_softmax_tile, args.encoder_pv_sdot, args.encoder_pv_gemm4x8, args.encoder_softmax_xnn, args.encoder_pquant_vector_store, args.decoder_cross_q1, args.encoder_pv_exact_accumulate)
    dependency_sources = {}
    kleidiai_options = []
    if args.kleidiai_source is not None:
        dependency = args.kleidiai_source.resolve(strict=True)
        if not (dependency / 'CMakeLists.txt').is_file():
            raise ValueError('KleidiAI source directory lacks CMakeLists.txt')
        # This explicit source override bypasses FetchContent's release download;
        # retain its actual complete identity, rather than claiming URL-hash proof.
        dependency_sources['kleidiai'] = {
            'path': str(dependency),
            'files_sha256': {str(p.relative_to(dependency)): sha(p)
                             for p in sorted(dependency.rglob('*'))
                             if p.is_file() and '.git' not in p.relative_to(dependency).parts}}
        kleidiai_options = ['-DGGML_CPU_KLEIDIAI=ON',
                            '-DFETCHCONTENT_SOURCE_DIR_KLEIDIAI=' + str(dependency)]
    commands = [
        ['cmake', '-S', str(here), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
         '-DGGML_SOURCE=' + str(ggml_source), '-DGGML_OPENMP=' + args.openmp,
         '-DMSTAR_ENCODER_QK_SDOT=' + ('ON' if args.encoder_qk_sdot else 'OFF'),
         '-DMSTAR_ENCODER_PV_SDOT=' + ('ON' if args.encoder_pv_sdot else 'OFF'),
         '-DMSTAR_ENCODER_Q8_CONV=' + str(args.encoder_q8_conv),
         '-DMSTAR_ENCODER_FFN_TILE=' + str(args.encoder_ffn_tile),
         '-DCMAKE_C_FLAGS=-fno-fast-math', '-DCMAKE_CXX_FLAGS=-fno-fast-math', *kleidiai_options],
        ['cmake', '--build', str(build), '--target', 'mstar_cpu_encoder', '-j', str(args.jobs)],
    ]
    report = {'preset': args.preset, 'whisper_source': str(source), 'whisper_revision': revision,
              'ggml_tree': tree, 'source_inventory_sha256': inventory_sha,
              'platform': platform.platform(), 'commands': commands, 'requested_openmp': args.openmp,
              'physical_source': str(ggml_source), 'physical_source_patches': patches,
              'dependency_sources': dependency_sources,
              'requested_overlays': {'shared_flash_pack': args.shared_flash_pack,
                                     'arm_simd_gemm': args.arm_simd_gemm,
                                     'encoder_qk_sdot': args.encoder_qk_sdot,
                                     'encoder_softmax_tile': args.encoder_softmax_tile,
                                     'encoder_pv_sdot': args.encoder_pv_sdot,
                                     'encoder_pv_gemm4x8': args.encoder_pv_gemm4x8,
                                     'encoder_softmax_xnn': args.encoder_softmax_xnn,
                                     'encoder_pquant_vector_store': args.encoder_pquant_vector_store,
                                     'decoder_cross_q1': args.decoder_cross_q1,
                                     'encoder_pv_exact_accumulate': args.encoder_pv_exact_accumulate,
                                     'encoder_q8_conv': args.encoder_q8_conv,
                                     'encoder_ffn_tile': args.encoder_ffn_tile}}
    if args.plan_only:
        report['status'] = 'PLAN_ONLY_NO_COMPILATION'
        (build / 'BUILD_PLAN.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return
    start = time.perf_counter()
    for i, command in enumerate(commands):
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        (build / f'command-{i}.stdout').write_text(result.stdout)
        (build / f'command-{i}.stderr').write_text(result.stderr)
        result.check_returncode()
    report['build_seconds'] = time.perf_counter() - start
    suffix = '.dylib' if platform.system() == 'Darwin' else '.so'
    library = build / ('libmstar_cpu_encoder' + suffix)
    report['cpu_ggml'] = {'library': str(library), 'sha256': sha(library)}
    report['library_bytes'] = library.stat().st_size
    shutil.copyfile(here / 'LICENSE.ggml', build / 'LICENSE.ggml')
    report['source_sha256'] = {str(p.relative_to(here)): sha(p) for p in here.iterdir() if p.is_file()}
    report['cmake_cache'] = (build / 'CMakeCache.txt').read_text()
    report['link_command'] = (build / 'CMakeFiles/mstar_cpu_encoder.dir/link.txt').read_text()
    report['compiler_flags'] = {str(p.relative_to(build)): p.read_text() for p in build.rglob('flags.make')}
    report['compiler_identity'] = {name: subprocess.check_output([name, '--version'], text=True)
                                   for name in ('cc', 'c++', 'cmake')}
    inspector = ['otool', '-L'] if platform.system() == 'Darwin' else ['ldd']
    report['dependencies'] = subprocess.check_output([*inspector, str(library)], text=True)
    exports = ['nm', '-gU', str(library)] if platform.system() == 'Darwin' else ['nm', '-D', '--defined-only', str(library)]
    report['exports'] = subprocess.check_output(exports, text=True)
    if platform.system() == 'Linux':
        exported = [line.split()[-1] for line in report['exports'].splitlines()]
        if any(name.startswith(('ggml_', 'whisper_')) for name in exported):
            raise RuntimeError('private ggml symbols escaped the component')
    library_handle = ctypes.CDLL(str(library))
    library_handle.cpu_encoder_abi.restype = ctypes.c_int
    if library_handle.cpu_encoder_abi() != 1:
        raise RuntimeError('CPU encoder ABI validation failed')
    (build / 'BUILD.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('whisper_revision', 'ggml_tree', 'build_seconds',
                      'cpu_ggml', 'library_bytes', 'dependencies', 'exports')}, indent=2))


if __name__ == '__main__':
    main()
