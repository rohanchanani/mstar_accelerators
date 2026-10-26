"""Talker-local BF16 router projection with one row-independent lowering."""
import torch
import triton
import triton.language as tl


@triton.jit
def _router_mm(x, weight, out, rows, sx0: tl.constexpr, sx1: tl.constexpr):
    # Fixed reduction and matrix-instruction geometry across all token counts.
    rm = tl.program_id(0) * 64 + tl.arange(0, 64)
    rn = tl.program_id(1) * 16 + tl.arange(0, 16)
    rk = tl.arange(0, 128)
    acc = tl.zeros((64, 16), dtype=tl.float32)
    for block in range(8):
        k = block * 128 + rk
        a = tl.load(x + rm[:, None] * sx0 + k[None, :] * sx1, rm[:, None] < rows, other=0)
        b = tl.load(weight + rn[None, :] * 1024 + k[:, None])
        acc += tl.dot(a, b, out_dtype=tl.float32)
    tl.store(out + rm[:, None] * 128 + rn[None, :], acc, rm[:, None] < rows)

@torch.library.custom_op("mstar_amd_a::router_mm", mutates_args=())
def router_mm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.ndim == 2 and x.shape[1] == 1024 and weight.shape == (128, 1024)
    assert weight.is_contiguous()
    out = torch.empty((x.shape[0], 128), device=x.device, dtype=x.dtype)
    if x.shape[0]:
        _router_mm[(triton.cdiv(x.shape[0], 64), 8)](
            x, weight, out, x.shape[0], x.stride(0), x.stride(1),
            num_warps=4, num_stages=2, matrix_instr_nonkdim=16, waves_per_eu=2, kpack=1,
        )
    return out

@router_mm.register_fake
def _router_mm_fake(x, weight):
    return x.new_empty((x.shape[0], weight.shape[0]))

from mstar.model.components.moe import (
    ParallelSparseMoeBlockWithSharedExpert,
    TopKRouter,
    _dispatch,
)
from torch.nn import functional as F


class ConsistentBF16Router(TopKRouter):
    """Use the same BF16 projection for ordinary launch and graph replay.

    Keep the original softmax, top-k and renormalization semantics. A fixed
    reduction geometry prevents token-count-dependent GEMM selection from
    changing this sensitive routing decision. The operation remains ordinary
    BF16 input/output with FP32 accumulation; it does not require capture.
    """
    def forward(self, hidden_states, router_states=None, *, consistent_prefill=False):
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        router_logits = router_mm(hidden_states, self.weight) if consistent_prefill else F.linear(hidden_states, self.weight)
        router_logits = F.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        routing_weights = router_top_value.to(router_logits.dtype)
        return routing_weights, router_indices, None

class PrefillConsistentMoe(ParallelSparseMoeBlockWithSharedExpert):
    """Standardize the initial-prefill router, preserving the sampling walks."""
    def forward(self, hidden_states, *, consistent_prefill=False):
        if not consistent_prefill:
            return super().forward(hidden_states)
        input_shape = hidden_states.shape
        flat = hidden_states.view(-1, hidden_states.shape[-1]).contiguous()
        shared = self.shared_expert(flat)
        routing_weights, selected_experts, _ = self.gate(flat, consistent_prefill=True)
        if self.comm_group.world_size == 1:
            routed = _dispatch(flat, self.experts.gate_up_proj, self.experts.down_proj,
                               self.num_experts, selected_experts, routing_weights, self.experts_backend)
        else:
            routed = self._dispatch_tp(flat, routing_weights, selected_experts)
        shared_gate = torch.sigmoid(self.shared_expert_gate(flat))
        return (routed + shared_gate * shared).view(input_shape)
