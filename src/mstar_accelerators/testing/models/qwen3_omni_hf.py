"""Stock Transformers Talker at the declared device and arithmetic precision.

Only checkpoint-name/layout adaptation and M* logical input interpretation are
supplied here. Attention, normalization, MoE and depth-predictor computation are
Transformers implementations, not submitted accelerator helpers. Dense caches
are reference state only; actual execution still goes through M*.
"""
from pathlib import Path
import json
import torch
from .qwen3_omni_oracle import TalkerOracle


class HuggingFaceTalkerOracle(TalkerOracle):
    def __init__(self, model, *, device, dtype):
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeTalkerModel, Qwen3OmniMoeTalkerCodePredictorModel,
        )
        self.config, self.device, self.dtype = model.config, torch.device(device), dtype
        raw = json.loads((Path(model.local_dir) / "config.json").read_text())
        hf = Qwen3OmniMoeConfig(**raw).talker_config
        source = model.logical_weights()
        self.weights = source.load(name for name in source.by_name if name.startswith("talker."))
        for name in self.weights:
            self.weights[name] = self.weights[name].detach().to(self.device, dtype)
        self.requests = {}
        self.modules = {}
        for prefix, cls, config in (
            ("talker.model", Qwen3OmniMoeTalkerModel, hf.text_config),
            ("talker.code_predictor.model", Qwen3OmniMoeTalkerCodePredictorModel, hf.code_predictor_config),
        ):
            config._attn_implementation = "eager"
            config._experts_implementation = "eager"
            with torch.device("meta"):
                module = cls(config)
            values = {}
            for name, parameter in module.named_parameters():
                logical = prefix + "." + name
                if name.endswith(".experts.gate_up_proj"):
                    stem = logical.removesuffix(".gate_up_proj")
                    value = torch.stack([torch.cat([self.weights[f"{stem}.{i}.{part}_proj.weight"]
                        for part in ("gate", "up")]) for i in range(config.num_experts)])
                elif name.endswith(".experts.down_proj"):
                    stem = logical.removesuffix(".down_proj")
                    value = torch.stack([self.weights[f"{stem}.{i}.down_proj.weight"]
                                         for i in range(config.num_experts)])
                else:
                    value = self.weights[logical]
                if value.shape != parameter.shape:
                    raise ValueError(f"HF reference checkpoint shape mismatch: {logical}")
                values[name] = value
            module.load_state_dict(values, assign=True, strict=True)
            module.rotary_emb = type(module.rotary_emb)(config, device=self.device)
            self.modules[prefix] = module.eval()
        # Packed HF experts own their new layout; discard the redundant unpacked
        # weight views. Projections/heads/embeddings remain shared, not copied.
        for name in tuple(self.weights):
            if ".layers." in name:
                del self.weights[name]

    def _backbone(self, x, prefix, cfg, history, start, *, moe):
        from transformers.cache_utils import DynamicCache
        module = self.modules[prefix]
        cache = DynamicCache(config=module.config)
        for layer, previous in enumerate(history):
            if previous is not None:
                k, v = previous
                cache.update(k.transpose(0, 1)[None], v.transpose(0, 1)[None], layer)
        positions = torch.arange(start, start + x.shape[0], device=self.device)[None]
        output = module(inputs_embeds=x[None], position_ids=positions,
                        past_key_values=cache, use_cache=True)
        history[:] = [(layer.keys[0].transpose(0, 1), layer.values[0].transpose(0, 1))
                      for layer in output.past_key_values.layers]
        return output.last_hidden_state[0]
