"""Measure matched plain F32/F16 attention from one actual native operand capture."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np


class Operands(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in ('q', 'k', 'v', 'mask')] + [
        (name, C.c_uint64) for name in ('q_bytes', 'kv_bytes', 'mask_bytes')] + [
        (name, C.c_int32) for name in ('threads', 'warmup', 'repetitions')]


class Sample(C.Structure):
    _fields_ = [(name, C.c_double) for name in ('f32_ms', 'f16_ms', 'encode_f16_ms', 'copy_f32_ms')]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--library-sha256', required=True)
    parser.add_argument('--capture-prefix', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repetitions', type=int, default=30)
    args = parser.parse_args()
    if sys.byteorder != 'little':
        raise ValueError('this captured-byte format requires a little-endian host')
    if args.report.exists():
        raise FileExistsError('preserve previous measurements; choose a fresh report')
    if sha(args.library) != args.library_sha256:
        raise ValueError('probe library SHA256 mismatch')
    prefix = str(args.capture_prefix.resolve())
    metadata_path = Path(prefix + '.json')
    metadata = json.loads(metadata_path.read_text())
    expected = dict(format='actual-cross-attention-v1', layer=0, queries=1, keys=1500,
                    width=384, heads=6, byte_order='little')
    if any(metadata.get(k) != v for k, v in expected.items()) or metadata['scale'] != .125:
        raise ValueError('capture is not the matched unscaled Q1/H6/D64/N1500 case')
    if metadata.get('storage') != 'f32':
        raise ValueError('representation experiment requires an actual F32 baseline capture')
    arrays, files = {}, {str(metadata_path): sha(metadata_path)}
    for name, size in (('q', 384), ('k', 1500 * 384), ('v', 1500 * 384), ('mask', 1500)):
        path = Path(prefix + f'.{name}.f32')
        data = np.fromfile(path, dtype='<f4')
        if data.size != size or (name != 'mask' and not np.isfinite(data).all()):
            raise ValueError('capture size/nonfinite operands differ: ' + name)
        arrays[name] = data
        files[str(path)] = sha(path)
    api = C.CDLL(str(args.library.resolve()))
    api.cpu_attention_probe_abi.restype = C.c_int
    if api.cpu_attention_probe_abi() != 1:
        raise ValueError('matched attention probe ABI unavailable')
    api.cpu_attention_probe.argtypes = [C.POINTER(Operands), C.POINTER(Sample),
        C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t]
    api.cpu_attention_probe.restype = C.c_int
    inputs = Operands(*(arrays[name].ctypes.data for name in ('q', 'k', 'v', 'mask')),
        arrays['q'].nbytes, arrays['k'].nbytes, arrays['mask'].nbytes,
        args.threads, args.warmup, args.repetitions)
    if not 1 <= args.repetitions <= 10000:
        raise ValueError('invalid repetition count')
    samples = (Sample * args.repetitions)()
    outputs = [np.empty(384, np.float32), np.empty(384, np.float32)]
    error = C.create_string_buffer(1024)
    if api.cpu_attention_probe(C.byref(inputs), samples, outputs[0].ctypes.data,
                               outputs[1].ctypes.data, error, len(error)):
        raise RuntimeError(error.value.decode())
    rows = [{name: getattr(sample, name) for name, _ in Sample._fields_} for sample in samples]
    if not all(np.isfinite(output).all() for output in outputs):
        raise ValueError('nonfinite matched operator output')
    delta = outputs[0].astype(np.float64) - outputs[1].astype(np.float64)
    report = dict(status='MEASURED', scope='matched operator only; not qualification or service gain',
        capture=metadata, capture_files_sha256=files, library=str(args.library.resolve()),
        library_sha256=args.library_sha256, threads=args.threads, warmup=args.warmup,
        repetitions=args.repetitions, order='attention and encoding/copy pairs each alternate order; encoding and writes separate',
        preparation='persistent K/V pre-encoded; same logical1500 keys, mask and scale in both paths',
        encoding_scope='full cross K and V conversion/write; single-thread helper; excludes allocation',
        copy_scope='full cross K and V F32 memcpy; excludes allocation',
        medians={name: statistics.median(row[name] for row in rows) for name, _ in Sample._fields_},
        samples=rows, output_difference=dict(max_abs=float(np.max(np.abs(delta))),
            rms=float(np.sqrt(np.mean(delta * delta))), scope='observation, no numerical verdict'),
        output_sha256=[hashlib.sha256(output.tobytes()).hexdigest() for output in outputs],
        public_scaled_padded_control='NOT RUN; different arithmetic and visibility, not this comparison')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], medians=report['medians'], report=str(args.report)), indent=2))


if __name__ == '__main__':
    main()
