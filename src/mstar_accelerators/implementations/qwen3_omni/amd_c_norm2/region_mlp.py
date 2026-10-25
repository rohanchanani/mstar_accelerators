"""BF16 code-predictor SwiGLU region kernels; scratch tuning implementation."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def gateup(X,W,A,M:tl.constexpr, BN:tl.constexpr,BK:tl.constexpr):
 m=tl.arange(0,16); n=tl.program_id(0)*BN+tl.arange(0,BN); k=tl.arange(0,BK)
 g=tl.full((16,BN),0,tl.float32);u=tl.full((16,BN),0,tl.float32)
 for off in range(tl.cdiv(1024,BK)):
  kk=off*BK+k
  x=tl.load(X+m[:,None]*1024+kk[None,:],m[:,None]<M,other=0)
  wg=tl.load(W+n[None,:]*1024+kk[:,None])
  wu=tl.load(W+(n[None,:]+3072)*1024+kk[:,None])
  g+=tl.dot(x,wg);u+=tl.dot(x,wu)
 gb=g.to(tl.bfloat16).to(tl.float32);ub=u.to(tl.bfloat16).to(tl.float32)
 s=(gb/(1+tl.exp(-gb))).to(tl.bfloat16).to(tl.float32)
 a=(s*ub).to(tl.bfloat16)
 tl.store(A+m[:,None]*3072+n[None,:],a,m[:,None]<M)

@tr.jit
def down(A,W,O,M:tl.constexpr, BN:tl.constexpr,BK:tl.constexpr,SK:tl.constexpr):
 m=tl.arange(0,16);n=tl.program_id(0)*BN+tl.arange(0,BN);k=tl.arange(0,BK)
 acc=tl.full((16,BN),0,tl.float32)
 for off in range(tl.cdiv(3072,SK*BK)):
  kk=(off*SK+tl.program_id(1))*BK+k
  a=tl.load(A+m[:,None]*3072+kk[None,:],(m[:,None]<M)&(kk[None,:]<3072),other=0)
  w=tl.load(W+n[None,:]*3072+kk[:,None],kk[:,None]<3072,other=0)
  acc+=tl.dot(a,w)
 tl.store(O+tl.program_id(1)*M*1024+m[:,None]*1024+n[None,:],acc,m[:,None]<M)

@tr.jit
def finish(P,O,M:tl.constexpr,SK:tl.constexpr,B:tl.constexpr,SB:tl.constexpr):
 j=tl.program_id(0)*B+tl.arange(0,B); s=tl.arange(0,SB)
 x=tl.load(P+s[:,None]*(M*1024)+j[None,:],(s[:,None]<SK)&(j[None,:]<M*1024),other=0)
 tl.store(O+j,tl.sum(x,axis=0),j<M*1024)

@tr.jit
def gateup_vec(X,W,A,M:tl.constexpr, BN:tl.constexpr):
 n=tl.program_id(0)*BN+tl.arange(0,BN);k=tl.arange(0,1024);m=tl.program_id(1)
 x=tl.load(X+m*1024+k).to(tl.float32)
 g=tl.sum(tl.load(W+n[:,None]*1024+k[None,:]).to(tl.float32)*x[None,:],1).to(tl.bfloat16).to(tl.float32)
 u=tl.sum(tl.load(W+(n[:,None]+3072)*1024+k[None,:]).to(tl.float32)*x[None,:],1).to(tl.bfloat16).to(tl.float32)
 s=(g/(1+tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
 tl.store(A+m*3072+n,s*u)

@tr.jit
def down_vec(A,W,O,M:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SK:tl.constexpr):
 n=tl.program_id(0)*BN+tl.arange(0,BN);k=tl.arange(0,BK);m=tl.program_id(1)
 acc=tl.full((BN,BK),0,tl.float32)
 for off in range(tl.cdiv(3072,BK)):
  kk=off*BK+k;a=tl.load(A+m*3072+kk,kk<3072,other=0).to(tl.float32)
  w=tl.load(W+n[:,None]*3072+kk[None,:],kk[None,:]<3072,other=0).to(tl.float32)
  acc+=w*a[None,:]
 tl.store(O+m*1024+n,tl.sum(acc,1))

def region(x,wg,wd,cfg):
 kind,gn,gk,gw,dn,dk,dw,sk=cfg;m=x.numel()//1024
 a=torch.empty((m,3072),device=x.device,dtype=x.dtype);o=torch.empty_like(x)
 if kind=='vec':
  gateup_vec[(3072//gn,m)](x,wg,a,m,gn,num_warps=gw,enable_fp_fusion=False)
  down_vec[(1024//dn,m)](a,wd,o,m,dn,dk,1,num_warps=dw,enable_fp_fusion=False)
 else:
  gateup[(3072//gn,)](x,wg,a,m,gn,gk,num_warps=gw,num_stages=2,matrix_instr_nonkdim=16,enable_fp_fusion=False)
  p=o if sk==1 else torch.empty((sk,m,1024),device=x.device,dtype=torch.float32)
  down[(1024//dn,sk)](a,wd,p,m,dn,dk,sk,num_warps=dw,num_stages=2,matrix_instr_nonkdim=16,enable_fp_fusion=False)
  if sk>1:finish[(tr.cdiv(m*1024,256),)](p,o,m,sk,256,tr.next_power_of_2(sk),num_warps=4)
 return o

@tr.jit
def fused_partial(X,WG,WD,P,M:tl.constexpr,BI:tl.constexpr,BO:tl.constexpr):
 i=tl.program_id(0)*BI+tl.arange(0,BI);m=tl.program_id(1); k=tl.arange(0,1024)
 x=tl.load(X+m*1024+k).to(tl.float32)
 gv=tl.load(WG+i[:,None]*1024+k[None,:]).to(tl.float32)
 uv=tl.load(WG+(i[:,None]+3072)*1024+k[None,:]).to(tl.float32)
 g=tl.sum(gv*x[None,:],1).to(tl.bfloat16).to(tl.float32)
 u=tl.sum(uv*x[None,:],1).to(tl.bfloat16).to(tl.float32)
 s=(g/(1+tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
 a=(s*u).to(tl.bfloat16).to(tl.float32)
 for part in range(1024//BO):
  o=part*BO+tl.arange(0,BO)
  w=tl.load(WD+o[:,None]*3072+i[None,:]).to(tl.float32)
  z=tl.sum(w*a[None,:],1)
  tl.store(P+tl.program_id(0)*M*1024+m*1024+o,z)

def full_region(x,wg,wd,bi=16,bo=256,warps=4):
 m=x.numel()//1024;p=torch.empty((3072//bi,m,1024),device=x.device,dtype=torch.float32);o=torch.empty_like(x)
 fused_partial[(3072//bi,m)](x,wg,wd,p,m,bi,bo,num_warps=warps,enable_fp_fusion=False)
 finish[(tr.cdiv(m*1024,16),)](p,o,m,3072//bi,16,tr.next_power_of_2(3072//bi),num_warps=4)
 return o

from mstar.model.components.distributed import ParallelGatedMLP
class PredictorRegionMLP(ParallelGatedMLP):
    region_enabled=True
    def forward(self,x):
        m=x.numel()//self.hidden_size
        if self.region_enabled and self.hidden_size==1024 and self.intermediate_size==3072 and m in (1,2,4,8):
            cfg=('vec',4,1024,4,4,4096,4,1) if m<=2 else ('dot',16,128,2,32,128,2,8)
            return region(x,self.gate_up_proj.weight,self.down_proj.weight,cfg)
        return super().forward(x)
