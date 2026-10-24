"""A's physical checkpoint packing and module preparation.

The logical model supplies named checkpoint values. This implementation chooses
fused QKV/expert layouts, Triton MoE, predictor-head stacking and device dtype.
"""
from __future__ import annotations

from copy import deepcopy
import torch
from mstar.model.loader.base import StackedParamRule


def stacked_parameter_rules(num_experts: int) -> list[StackedParamRule]:
    rules = []
    for i in range(num_experts):
        for source, target, shard in (
            ("gate", "gate_up", "gate"),
            ("up", "gate_up", "up"),
            ("down", "down", "down"),
        ):
            rules.append(StackedParamRule(
                target_suffix=f".experts.{target}_proj",
                source_suffix=f".experts.{source}_proj.__expert{i}__.weight",
                shard_id=f"{shard}:{i}",
            ))
    # Longer expert suffixes must precede the dense projection suffixes.
    rules.extend([
        StackedParamRule(".gate_up_proj", ".gate_proj", 0),
        StackedParamRule(".gate_up_proj", ".up_proj", 1),
        StackedParamRule(".qkv_proj", ".q_proj", "q"),
        StackedParamRule(".qkv_proj", ".k_proj", "k"),
        StackedParamRule(".qkv_proj", ".v_proj", "v"),
    ])
    return rules


def remap_checkpoint_name(name: str) -> str | None:
    import re
    if "rotary_emb" in name:
        return None
    name = name.replace("block_sparse_moe.", "mlp.")
    match = re.match(r"(.*)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", name)
    if match:
        prefix, expert_idx, projection = match.groups()
        return f"{prefix}.experts.{projection}.__expert{expert_idx}__.weight"
    return name


def prepare_program(binding, resources):
    from mstar.model.components.moe import prepare_moe_weights
    from mstar.model.loader import load_hf_weights
    from .model import Qwen3OmniCodePredictor, Qwen3OmniTalkerModel
    from .program import TalkerProgram

    config = deepcopy(binding.model.config)
    config.talker_text.moe_backend = "triton"
    values = binding.model.logical_weights().load()
    rules = stacked_parameter_rules(config.talker_text.num_experts)
    with torch.device("meta"):
        talker = Qwen3OmniTalkerModel(config, comm_group=binding.tp_group)
        predictor = Qwen3OmniCodePredictor(config)
    for module, prefix, exclude in (
        (talker, "talker.", ("talker.code_predictor.", "talker.conditioning.")),
        (predictor, "talker.code_predictor.", ()),
    ):
        module.to(dtype=binding.dtype).to_empty(device=binding.device)
        weights = (
            (name.removeprefix(prefix), value.to(device=binding.device, dtype=binding.dtype))
            for name, value in values.items()
            if name.startswith(prefix) and not name.startswith(exclude)
        )
        loaded = load_hf_weights(
            module, weights, stacked_params=rules,
            name_remapper=remap_checkpoint_name,
        )
        missing = set(dict(module.named_parameters())) - loaded
        if missing:
            raise ValueError(f"AMD Talker program did not initialize parameters: {sorted(missing)}")
        module.eval()
    prepare_moe_weights(talker)
    predictor.consolidate_stacked_weights()
    tts_embeddings = torch.stack([
        values[f"talker.conditioning.tts_{name}"] for name in ("pad", "bos", "eos")
    ]).to(device=binding.device, dtype=binding.dtype)
    program = TalkerProgram(talker, predictor, tts_embeddings, config)
    program.bind_resources(resources)
    return program.eval()
