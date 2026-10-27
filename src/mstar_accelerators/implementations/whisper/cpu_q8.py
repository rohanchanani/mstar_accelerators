"""Shared immutable ARM readout preparation in the existing binding owner."""
import torch

from mstar_accelerators.backends.weights import prepare_weights
from mstar_accelerators.implementations.whisper.pi_v3d.q8 import pack_q8_0
from mstar_accelerators.implementations.whisper.torch import column_major_weights


def prepared_weights(binding, weight_prefix):
    # Ordinary module identity is shared by path-loaded and canonically imported
    # coloring books. Mutable request/KV/sampling state is never stored here.
    key = (prepared_weights, weight_prefix)
    if key not in binding.prepared:
        weights = prepare_weights(binding, weight_prefix, dtype=torch.float32)
        embedding, projection = "decoder.embed_tokens.weight", "decoder.proj_out.weight"
        source_body = {name: weight for name, weight in weights.items() if name not in (embedding, projection)}
        body = column_major_weights(binding, source_body) | {embedding: weights[embedding]}
        packed = pack_q8_0(weights[projection])
        suppressed = {walk: torch.tensor(binding.submodule.config.suppressed_tokens(walk), dtype=torch.int64, device=binding.device)
                      for walk in ("prefill", "decode")}
        # Keep mapping identities and immutable representations in the existing
        # node owner. The tied embedding stays F32 without a column readout copy.
        binding.prepared[key] = (source_body, body, packed, suppressed, weights[projection].shape)
    return binding.prepared[key]
