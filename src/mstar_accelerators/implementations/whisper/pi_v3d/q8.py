"""Explicit immutable Q8_0 representation for the optional decoder readout."""
from __future__ import annotations

from dataclasses import dataclass
import sys

import torch

from mstar_accelerators.backends.graph import NativeGraphProvider


READOUT_WEIGHT = "decoder.proj_out.weight"
READOUT_Q8_0 = READOUT_WEIGHT + ".q8_0"


def pack_q8_0(weight: torch.Tensor) -> torch.Tensor:
    """Pack rows as ggml block_q8_0: FP16 scale, then 32 signed bytes.

    Match ggml-quants.c quantize_row_q8_0_ref: scale and reciprocal use
    FP32, integer rounding is halfway away from zero, and only the stored
    scale is converted to FP16. The result is an exact flat I32 byte carrier.
    This is preparation arithmetic, never a request-time compute fallback.
    """
    if sys.byteorder != "little":
        raise ValueError("Q8_0 readout packing requires little-endian storage")
    if weight.device.type != "cpu" or weight.dtype != torch.float32 or weight.ndim != 2:
        raise ValueError("Q8_0 packing requires a CPU FP32 weight matrix")
    rows, width = weight.shape
    if rows <= 0 or width <= 0 or width % 32 or (rows * (width // 32) * 34) % 4:
        raise ValueError("Q8_0 extent must contain whole 32-value blocks and I32 words")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("Q8_0 packing requires finite logical weights")
    blocks = weight.contiguous().reshape(-1, 32)
    scale = blocks.abs().amax(dim=1) / 127.0
    inverse = torch.where(scale != 0, scale.reciprocal(), 0.0)
    scaled = blocks * inverse[:, None]
    rounded = scaled.round()
    # torch.round uses ties-to-even; C roundf uses halfway away from zero.
    magnitude = scaled.abs()
    ties = (magnitude - magnitude.floor()) == 0.5
    rounded = torch.where(ties, scaled.trunc() + scaled.sign(), rounded)
    packed = torch.empty((blocks.shape[0], 34), dtype=torch.uint8)
    packed[:, :2] = scale.to(torch.float16).view(torch.uint8).reshape(-1, 2)
    packed[:, 2:] = rounded.to(torch.int8).view(torch.uint8)
    return packed.reshape(-1).view(torch.int32)


@dataclass(frozen=True)
class _ReadoutRepresentation:
    logical_name: str
    format: str
    device: str


class Q8ReadoutProvider:
    """Keep the readout copy in the existing binding's prepared-weight owner."""

    def __init__(self, binding):
        key = _ReadoutRepresentation(READOUT_WEIGHT, "ggml-q8_0-v1", str(binding.device))
        if key not in binding.prepared:
            logical = binding.model.logical_weights().load((READOUT_WEIGHT,))[READOUT_WEIGHT]
            packed = pack_q8_0(logical.to(device="cpu", dtype=torch.float32))
            binding.prepared[key] = {READOUT_Q8_0: packed.to(binding.device)}
        self.constants = binding.prepared[key]
        self.native = NativeGraphProvider()

    def prepare(self, builder, constants, *, slots):
        return self.native.prepare(builder, dict(constants) | self.constants, slots=slots)


def q8_readout(g, config, last):
    """Quantize only the last query, then explicitly dot the prepared weights."""
    weight = g.tensor(READOUT_Q8_0, (config.d_model, config.vocab_size), dtype="q8_0")
    quantized = g.quantize_q8_1_x4(last)
    return g.q8_0_dot_x4(weight, quantized)
