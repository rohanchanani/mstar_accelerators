"""Split-K down reduction with the following BF16 residual/RMS boundary."""
import torch
import triton
import triton.language as tl
from .region_mlp import gateup,down

@triton.jit
def finish_residual_norm(P,R,W,Y,S,M:tl.constexpr,EPS:tl.constexpr):
    row=tl.program_id(0)
    k=tl.arange(0,8)
    n=tl.arange(0,1024)
    part=tl.load(P+k[:,None]*(M*1024)+row*1024+n[None,:])
    delta=tl.sum(part,axis=0).to(tl.bfloat16).to(tl.float32)
    residual=tl.load(R+row*1024+n).to(tl.float32)
    summed=(residual+delta).to(tl.bfloat16)
    rounded=summed.to(tl.float32)
    variance=tl.sum(rounded*rounded,axis=0)/1024
    norm=rounded*tl.rsqrt(variance+EPS)
    weight=tl.load(W+n).to(tl.float32)
    tl.store(Y+row*1024+n,norm*weight)
    tl.store(S+row*1024+n,summed)

def predictor_mlp_next_norm(x,wg,wd,residual,norm_weight,eps):
    m=x.numel()//1024
    activation=torch.empty((m,3072),device=x.device,dtype=x.dtype)
    partial=torch.empty((8,m,1024),device=x.device,dtype=torch.float32)
    normalized=torch.empty_like(x)
    summed=torch.empty_like(x)
    gateup[(192,)](x,wg,activation,m,16,128,num_warps=2,num_stages=2,matrix_instr_nonkdim=16,enable_fp_fusion=False)
    down[(32,8)](activation,wd,partial,m,32,128,8,num_warps=2,num_stages=2,matrix_instr_nonkdim=16,enable_fp_fusion=False)
    finish_residual_norm[(m,)](partial,residual,norm_weight,normalized,summed,m,eps,num_warps=4,enable_fp_fusion=False)
    return normalized,summed
