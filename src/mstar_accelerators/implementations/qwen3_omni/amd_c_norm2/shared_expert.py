"""BF16-preserving TP1 Talker shared expert and routed-output epilogue.

The router and routed experts keep their existing execution. Two kernels own
only the standard dense shared branch and its sigmoid-weighted addition.
Unsupported modules, layouts and prefill paths use the inherited implementation.
"""

import torch
import triton as tr
import triton.language as tl
from mstar.model.components.distributed import ParallelGatedMLP
from mstar.model.components.distributed.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from mstar.model.components.moe import _dispatch

from ..amd_a.router_consistency import PrefillConsistentMoe


@tr.jit
def _prepare(X, W, SG, A, G, BN: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, 1024)
    m = tl.program_id(1)
    x = tl.load(X + m * 1024 + k).to(tl.float32)
    g = (
        tl.sum(
            tl.load(W + n[:, None] * 1024 + k[None, :]).to(tl.float32) * x[None, :], 1
        )
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    u = (
        tl.sum(
            tl.load(W + (n[:, None] + 768) * 1024 + k[None, :]).to(tl.float32)
            * x[None, :],
            1,
        )
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    s = (g / (1 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(A + m * 768 + n, (s * u).to(tl.bfloat16))
    if tl.program_id(0) == 0:
        gate = (
            tl.sum(tl.load(SG + k).to(tl.float32) * x, 0).to(tl.bfloat16).to(tl.float32)
        )
        tl.store(G + m, tl.sigmoid(gate).to(tl.bfloat16))


@tr.jit
def _finish(A, W, G, ROUTED, O, BN: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, 1024)
    m = tl.program_id(1)
    a = tl.load(A + m * 768 + k, k < 768, other=0).to(tl.float32)
    w = tl.load(W + n[:, None] * 768 + k[None, :], k[None, :] < 768, other=0).to(
        tl.float32
    )
    down = tl.sum(w * a[None, :], 1).to(tl.bfloat16).to(tl.float32)
    gated = (down * tl.load(G + m).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    routed = tl.load(ROUTED + m * 1024 + n).to(tl.float32)
    tl.store(O + m * 1024 + n, (routed + gated).to(tl.bfloat16))


class SharedRegionMoe(PrefillConsistentMoe):
    shared_region_enabled = True

    def _eligible(self, x):
        shared = self.shared_expert
        if not self.shared_region_enabled or self.comm_group.world_size != 1:
            return False
        if (
            type(shared) is not ParallelGatedMLP
            or shared.act is not torch.nn.functional.silu
        ):
            return False
        gu, down, sg = shared.gate_up_proj, shared.down_proj, self.shared_expert_gate
        if (
            type(gu) is not MergedColumnParallelLinear
            or type(down) is not RowParallelLinear
            or type(sg) is not torch.nn.Linear
        ):
            return False
        if gu.comm_group.world_size != 1 or down.comm_group.world_size != 1:
            return False
        if any(mod.bias is not None for mod in (gu, down, sg)):
            return False
        if x.shape[-1] != 1024 or x.numel() // 1024 not in (1, 2, 4, 8):
            return False
        if not x.is_cuda or not torch.version.hip or torch.is_grad_enabled():
            return False
        if not torch.cuda.get_device_properties(x.device).gcnArchName.startswith(
            "gfx942"
        ):
            return False
        if (
            tuple(gu.weight.shape) != (1536, 1024)
            or tuple(down.weight.shape) != (1024, 768)
            or tuple(sg.weight.shape) != (1, 1024)
        ):
            return False
        return all(
            t.dtype == torch.bfloat16 and t.device == x.device and t.is_contiguous()
            for t in (x, gu.weight, down.weight, sg.weight)
        )

    def forward(self, hidden_states, *, consistent_prefill=False):
        if consistent_prefill or not self._eligible(hidden_states):
            return super().forward(hidden_states, consistent_prefill=consistent_prefill)
        shape = hidden_states.shape
        flat = hidden_states.view(-1, 1024)
        m = flat.shape[0]
        intermediate = torch.empty((m, 768), device=flat.device, dtype=flat.dtype)
        gate = torch.empty((m,), device=flat.device, dtype=flat.dtype)
        _prepare[(384, m)](
            flat,
            self.shared_expert.gate_up_proj.weight,
            self.shared_expert_gate.weight,
            intermediate,
            gate,
            2,
            num_warps=1,
            enable_fp_fusion=False,
        )
        routing_weights, selected_experts, _ = self.gate(flat, None)
        routed = _dispatch(
            flat,
            self.experts.gate_up_proj,
            self.experts.down_proj,
            self.num_experts,
            selected_experts,
            routing_weights,
            self.experts_backend,
        )
        # Alternative routed backends need not return our dense BF16 layout.
        # Reuse their result; never repeat or alter the routing computation.
        if (
            routed.dtype != flat.dtype
            or routed.device != flat.device
            or routed.shape != flat.shape
            or not routed.is_contiguous()
        ):
            shared = self.shared_expert(flat)
            shared_gate = torch.sigmoid(self.shared_expert_gate(flat))
            return (routed + shared_gate * shared).view(shape)
        out = torch.empty_like(flat)
        _finish[(512, m)](
            intermediate,
            self.shared_expert.down_proj.weight,
            gate,
            routed,
            out,
            2,
            num_warps=1,
            enable_fp_fusion=False,
        )
        return out.view(shape)
