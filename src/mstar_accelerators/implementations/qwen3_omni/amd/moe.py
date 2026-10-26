"""Select physical expert kernels within this acceleration's owned module tree."""
import torch
from mstar.utils.fused_moe.backend import TritonFusedMoE
from mstar.utils.fused_moe.config import MoEProblem

def select_expert_kernels(module):
    if not torch.version.hip or not torch.cuda.is_available():
        return
    if torch.cuda.get_device_properties(next(module.parameters()).device).gcnArchName.split(":")[0] != "gfx942":
        return
    from .talker_moe import SmallTokenTalkerMoE
    from .thinker_moe import SmallTokenThinkerMoE
    for child in module.modules():
        backend = getattr(child, "experts_backend", None)
        if not isinstance(backend, TritonFusedMoE):
            continue
        problem = backend.problem
        if problem == MoEProblem(128, 384, 1024, 6):
            child.experts_backend = SmallTokenTalkerMoE(problem)
        elif problem == MoEProblem(128, 768, 2048, 8):
            child.experts_backend = SmallTokenThinkerMoE(problem)
