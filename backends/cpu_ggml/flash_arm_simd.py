"""Use upstream's existing ARM GEMM block with the standard NEON macro.

The pinned header checks only the legacy __ARM_NEON__ spelling. GCC 14.2 on
the target defines __ARM_NEON instead, selecting the generic 2x2 vector block.
No arithmetic, tile size, scratch layout or graph ABI is changed.
"""
import argparse
import hashlib
from pathlib import Path

HEADER_SHA256 = 'e82a07445059ec4a4c7d8dd75a2663d459025274d817be3f83c1f5989a79208d'


def header_overlay(text):
    old = '#if defined(__AVX512F__) || defined (__ARM_NEON__)'
    if text.count(old) != 1:
        raise ValueError('pinned SIMD GEMM ARM selection anchor differs')
    return text.replace(old, old + ' || defined (__ARM_NEON)', 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ggml-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = args.ggml_source / 'src/ggml-cpu/simd-gemm.h'
    if hashlib.sha256(original.read_bytes()).hexdigest() != HEADER_SHA256:
        raise ValueError('SIMD GEMM source differs from pinned upstream')
    with args.output.open('x') as output:
        output.write(header_overlay(original.read_text()))


if __name__ == '__main__':
    main()
