"""Independent eager CPU interpretation of the logical Talker computation.

No accelerated numerical model, packed parameter, kernel, resource, or runner is
used here. Ordinary PyTorch linear algebra evaluates named logical weights with
dense per-request KV history. This is a correctness oracle, never a serving
fallback. Float32 is the default; full checkpoint weights consume their logical
parameter count times four bytes, plus temporary loading storage and short KV
histories. Evaluate short requests when qualifying a full checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from mstar.model.qwen3_omni.results import TalkerResult


@dataclass(frozen=True)
class OracleStep:
    result: TalkerResult | None
    kv_length: int
    position: int
    sampler_offsets: tuple[int, int]
    seen_tokens: frozenset[int]
    talker_logits: torch.Tensor | None = None
    predictor_logits: tuple[torch.Tensor, ...] = ()


@dataclass
class _Request:
    kv: list
    length: int = 0
    seen: set[int] = field(default_factory=set)
    talker_draws: int = 0
    predictor_draws: int = 0


class TalkerOracle:
    """One independent semantic state per request handle.

    ``step`` accepts one already interpreted logical NodeInputs. Optional
    ``teacher_codes`` supplies a shared depth history. Qualification
    must score each supplied code against the corresponding reference logits;
    supplying history alone does not constitute a passing comparison.
    """

    def __init__(self, model, *, dtype: torch.dtype = torch.float32):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("The portable oracle supports CPU float32 or float64")
        self.config = model.config
        self.dtype = dtype
        self.device = torch.device("cpu")
        source = model.logical_weights()
        self.weights = source.load(name for name in source.by_name if name.startswith("talker."))
        # Replace one entry at a time so the source dictionary does not retain
        # every BF16 tensor alongside a complete second FP32 dictionary.
        for name in self.weights:
            self.weights[name] = self.weights[name].detach().to(device="cpu", dtype=dtype)
        self.requests: dict[Any, _Request] = {}

    def reset(self, request_id) -> None:
        self.requests.pop(request_id, None)

    def kv(self, request_id) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Logical K/V tensors per layer, each [positions, KV heads, head dim]."""
        return tuple(self.requests[request_id].kv)

    def _linear(self, x, prefix, *, bias=False):
        return F.linear(x, self.weights[prefix + ".weight"],
                        self.weights[prefix + ".bias"] if bias else None)

    def _norm(self, x, prefix, eps):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * self.weights[prefix + ".weight"]

    def _project(self, x, kind):
        prefix = "talker." + kind + "_projection"
        return self._linear(F.silu(self._linear(x, prefix + ".linear_fc1", bias=True)),
                            prefix + ".linear_fc2", bias=True)

    def _mlp(self, x, prefix):
        return self._linear(F.silu(self._linear(x, prefix + ".gate_proj")) *
                            self._linear(x, prefix + ".up_proj"), prefix + ".down_proj")

    def _rope(self, x, positions, theta):
        half = x.shape[-1] // 2
        inverse = theta ** (-torch.arange(half, dtype=self.dtype) / half)
        angles = positions.to(self.dtype)[:, None] * inverse[None, :]
        first, second = x.chunk(2, dim=-1)
        cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
        return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)

    def _attention(self, x, prefix, cfg, previous, positions, theta):
        tokens = x.shape[0]
        q = self._linear(x, prefix + ".q_proj").reshape(tokens, cfg.num_attention_heads, cfg.head_dim)
        k = self._linear(x, prefix + ".k_proj").reshape(tokens, cfg.num_key_value_heads, cfg.head_dim)
        v = self._linear(x, prefix + ".v_proj").reshape(tokens, cfg.num_key_value_heads, cfg.head_dim)
        q = self._rope(self._norm(q, prefix + ".q_norm", cfg.rms_norm_eps), positions, theta)
        k = self._rope(self._norm(k, prefix + ".k_norm", cfg.rms_norm_eps), positions, theta)
        if previous is not None:
            k, v = torch.cat((previous[0], k)), torch.cat((previous[1], v))
        repeats = cfg.num_attention_heads // cfg.num_key_value_heads
        full_k = k.repeat_interleave(repeats, dim=1).transpose(0, 1)
        full_v = v.repeat_interleave(repeats, dim=1).transpose(0, 1)
        scores = (q.transpose(0, 1) @ full_k.transpose(-2, -1)) * cfg.head_dim ** -0.5
        allowed = torch.arange(k.shape[0])[None, :] <= positions[:, None]
        scores = scores.masked_fill(~allowed[None, :, :], float("-inf"))
        mixed = (scores.softmax(dim=-1) @ full_v).transpose(0, 1).reshape(tokens, -1)
        return self._linear(mixed, prefix + ".o_proj"), (k, v)

    def _moe(self, x, prefix):
        cfg = self.config.talker_text
        routing = self._linear(x, prefix + ".gate").softmax(dim=-1)
        amounts, experts = routing.topk(cfg.num_experts_per_tok, dim=-1)
        if cfg.norm_topk_prob:
            amounts = amounts / amounts.sum(dim=-1, keepdim=True)
        routed = torch.zeros_like(x)
        for expert in torch.unique(experts).tolist():
            rows, slots = torch.where(experts == expert)
            values = self._mlp(x[rows], f"{prefix}.experts.{expert}")
            routed.index_add_(0, rows, values * amounts[rows, slots, None])
        shared = self._mlp(x, prefix + ".shared_expert")
        return routed + self._linear(x, prefix + ".shared_expert_gate").sigmoid() * shared

    def _backbone(self, x, prefix, cfg, history, start, *, moe):
        positions = torch.arange(start, start + x.shape[0])
        theta = 1_000_000.0 if moe else cfg.rope_theta
        for layer in range(cfg.num_hidden_layers):
            stem = f"{prefix}.layers.{layer}"
            attention, history[layer] = self._attention(
                self._norm(x, stem + ".input_layernorm", cfg.rms_norm_eps),
                stem + ".self_attn", cfg, history[layer], positions, theta,
            )
            x = x + attention
            normed = self._norm(x, stem + ".post_attention_layernorm", cfg.rms_norm_eps)
            x = x + (self._moe(normed, stem + ".mlp") if moe else self._mlp(normed, stem + ".mlp"))
        return self._norm(x, prefix + ".norm", cfg.rms_norm_eps)

    def _materialize(self, walk, inputs):
        raw = inputs.tensor_inputs["thinker_states"].detach().to(device=self.device, dtype=self.dtype)
        if walk == "talker_prefill":
            mask = inputs.tensor_inputs["multimodal_mask"].detach().to(device=self.device, dtype=torch.bool)
            hidden = torch.empty((raw.shape[0], self.config.talker_hidden_size), dtype=self.dtype, device=self.device)
            for selected, kind in ((~mask, "text"), (mask, "hidden")):
                if selected.any():
                    hidden[selected] = self._project(raw[selected], kind)
            return hidden
        if walk == "talker_last_prefill":
            if raw.shape[0] != 6:
                raise ValueError("The logical assistant prefix has six rows")
            text = torch.cat((self._project(raw[:1], "text").expand(4, -1),
                              self._project(raw[4:5], "text"), self._project(raw[5:], "text")))
            ids = inputs.tensor_inputs["codec_ids"].detach().to(self.device)
            return F.embedding(ids, self.weights["talker.model.codec_embedding.weight"]) + text
        return inputs.input_embeds.detach().to(device=self.device, dtype=self.dtype) + self._project(raw, "text")

    @torch.no_grad()
    def step(self, walk, request_id, inputs, *, repetition_penalty=1.0, teacher_codes=None) -> OracleStep:
        if walk not in ("talker_prefill", "talker_last_prefill", "talker_decode"):
            raise ValueError(f"Unknown logical Talker walk: {walk}")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        cfg, cp = self.config.talker_text, self.config.code_predictor
        state = self.requests.setdefault(request_id, _Request([None] * cfg.num_hidden_layers))
        embeddings = self._materialize(walk, inputs)
        hidden = self._backbone(embeddings, "talker.model", cfg, state.kv, state.length, moe=True)
        state.length += embeddings.shape[0]
        if walk == "talker_prefill":
            return OracleStep(None, state.length, state.length,
                              (state.talker_draws, state.predictor_draws), frozenset(state.seen))

        logits = self._linear(hidden[-1:], "talker.codec_head")[0]
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        eos = self.config.talker.codec_eos_token_id
        suppressed = torch.arange(max(0, cfg.vocab_size - 1024), cfg.vocab_size, device=self.device)
        logits[suppressed[suppressed != eos]] = float("-inf")
        if state.seen and repetition_penalty != 1.0:
            seen = torch.tensor(sorted(state.seen), device=self.device)
            values = logits[seen]
            logits[seen] = torch.where(values > 0, values / repetition_penalty, values * repetition_penalty)
        groups = self.config.num_code_groups
        teacher = None if teacher_codes is None else torch.as_tensor(teacher_codes, dtype=torch.long, device=self.device).reshape(-1)
        if teacher is not None and teacher.numel() != groups:
            raise ValueError(f"teacher_codes must have {groups} entries")
        codes = torch.empty(groups, dtype=torch.long, device=self.device)
        codes[0] = logits.argmax() if teacher is None else teacher[0]
        embedding = F.embedding(codes[:1], self.weights["talker.model.codec_embedding.weight"])
        accumulator = embedding.float().clone()
        depth_kv = [None] * cp.num_hidden_layers
        self._backbone(hidden[-1:], "talker.code_predictor.model", cp, depth_kv, 0, moe=False)
        depth_logits = []
        for group in range(1, groups):
            depth_hidden = self._backbone(embedding, "talker.code_predictor.model", cp, depth_kv, group, moe=False)
            candidate_logits = self._linear(depth_hidden, f"talker.code_predictor.lm_head.{group - 1}")[0]
            depth_logits.append(candidate_logits)
            codes[group] = candidate_logits.argmax() if teacher is None else teacher[group]
            embedding = F.embedding(codes[group:group + 1],
                                    self.weights[f"talker.code_predictor.model.codec_embedding.{group - 1}.weight"])
            accumulator.add_(embedding.float())
        state.seen.add(int(codes[0]))
        state.talker_draws += 1
        state.predictor_draws += groups - 1
        result = TalkerResult(codec_tokens=codes[None, :], codec_embeds=accumulator,
                              tokens=codes[:1])
        return OracleStep(result, state.length, state.length,
                          (state.talker_draws, state.predictor_draws), frozenset(state.seen),
                          logits, tuple(depth_logits))
