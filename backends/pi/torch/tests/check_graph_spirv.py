"""Inspect real generated graph shaders without requiring a GPU.

Checks the physical ABI of every selected non-matrix managed kernel family:
no 64-bit integer arithmetic, F16 conversion, or fixed workgroup over 256.
This is a static artifact check, not numerical or hardware qualification.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import struct


def inspect(path: Path) -> dict:
    raw = path.read_bytes()
    words = struct.unpack(f"<{len(raw)//4}I", raw)
    assert words[0] == 0x07230203
    integers, floats, capabilities = set(), set(), set()
    fixed_local_sizes = []
    conversions = 0
    extended_multiplies = 0
    i = 5
    while i < len(words):
        count, op = words[i] >> 16, words[i] & 65535
        assert count and i + count <= len(words)
        args = words[i+1:i+count]
        if op == 17: capabilities.add(args[0])  # OpCapability
        if op == 21: integers.add(args[1])     # OpTypeInt
        if op == 22: floats.add(args[1])       # OpTypeFloat
        if op == 115: conversions += 1        # OpFConvert
        if op == 151: extended_multiplies += 1  # OpUMulExtended
        if op == 16 and args[1] == 17: fixed_local_sizes.append(list(args[2:5]))  # LocalSize
        i += count
    assert i == len(words)
    return {"shader": path.name, "integer_widths": sorted(integers), "float_widths": sorted(floats),
            "capabilities": sorted(capabilities), "float_conversions": conversions,
            "extended_multiplies": extended_multiplies,
            "fixed_local_sizes": fixed_local_sizes}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("spv_directory",type=Path);args=parser.parse_args()
    records=[]
    for name in ("cpy_i32_i64", "cpy_i64_i32", "cpy_i64", "argmax_f32", "im2col_f32",
                 "managed_cpy_32", "managed_norm_f32", "managed_gelu_erf_f32",
                 "managed_get_rows_f32", "managed_set_rows_f32_i32", "managed_set_rows_f32_i64",
                 "managed_categorical_f32",
                 "add_f32_f32_f32", "mul_f32_f32_f32", "scale_f32", "soft_max_f32"):
        result=inspect(args.spv_directory/f"{name}.spv")
        assert 64 not in result["integer_widths"], result
        assert 11 not in result["capabilities"], result  # Int64
        assert 16 not in result["float_widths"], result
        assert result["float_conversions"] == 0, result
        for x,y,z in result["fixed_local_sizes"]:
            assert x <= 256 and x*y*z <= 256, result
        if name == "managed_categorical_f32":
            assert result["extended_multiplies"] >= 2, result
            assert result["fixed_local_sizes"] == [[256, 1, 1]], result
        records.append(result)
    print(json.dumps({"status":"PASS", "scope":"actual graph shader ABI", "records":records},sort_keys=True))


if __name__ == "__main__":main()
