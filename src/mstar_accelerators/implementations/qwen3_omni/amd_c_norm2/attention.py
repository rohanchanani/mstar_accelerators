"""AMD A standard 1-D attention, using selected runtime resource primitives."""
from __future__ import annotations

import torch
from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import ParallelAttention

class Qwen3OmniAttention(ParallelAttention):
    """A's QK-normalized attention with standard one-dimensional RoPE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float = 1_000_000.0,
        rms_norm_eps: float = 1e-6,
        use_mrope: bool = False,
        comm_group: CommGroup | None = None,
        attn_key: str = "attn",
        kv_key: str = "kv",
        pos_key: str | None = "rope"
    ):
        super().__init__(
            comm_group=comm_group,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qkv_bias=False,
            o_bias=False,
            qk_norm=True,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            attn_key=attn_key,
            kv_key=kv_key,
            pos_key=pos_key

        )
        if use_mrope:
            raise ValueError("The AMD Talker program uses standard 1-D RoPE")

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        q, k, v = self._project_qkv(hidden_states)
        q, k = self._apply_qk_norm(q, k)

        q, k = self._apply_rope(q, k, self.attend.label)

        attn_output = self.attend(q, k, v)
        attn_output = attn_output.reshape(num_tokens, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)
