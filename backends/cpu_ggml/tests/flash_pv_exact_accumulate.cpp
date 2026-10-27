// Physical operator controls only; no replacement model oracle/policy.
#include "common.h"
#include "vec.h"
#include "flash_pv_gemm.h"
#include "flash_pv_exact_accumulate.h"
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
namespace {
void old_acc(float *acc,const block_q8_0x4 *v,const block_q8_0x4 *p){alignas(64) float product[4096];mstar_pv_gemm4x8(64,product,64,v,p,64,64);ggml_vec_acc_f32(4096,acc,product);}
int quant(const block_q8_0x4*p,int row,int k){return p[row/4*2+k/32].qs[k%32/4*16+row%4*4+k%4];}
float scale(const block_q8_0x4*p,int row,int block){return GGML_CPU_FP16_TO_FP32(p[row/4*2+block].d[row%4]);}
void fma_controls(){
 constexpr size_t guard=16;uint64_t checked=0,changed=0,fallback=0;double max_delta=0,max_error=0;
 for(int kind=0;kind<24;++kind){
  std::array<block_q8_0x4,32> pp{},vv{};
  const float scales[]={0.f,-0.f,1.f,-1.f,65504.f,0x1p-24f,0x1p-14f,0.03125f};
  for(int i=0;i<32;++i){for(int j=0;j<4;++j){pp[i].d[j]=ggml_fp32_to_fp16(scales[(i+j+kind)%8]);vv[i].d[j]=ggml_fp32_to_fp16(scales[(i*3+j+kind+1)%8]);}
   for(int j=0;j<128;++j){pp[i].qs[j]=int8_t((kind+j*17+i*29)%256-128);vv[i].qs[j]=int8_t((kind*5+j*13+i*23)%256-128);}}
  std::array<float,4096+2*guard> a,b;a.fill(12345.25f);
  for(int i=0;i<4096;++i)a[guard+i]=kind%3==0?float((i*17+kind)%101-50)/31.f:kind%3==1?(i%2?1e20f:-1e20f):(i%2?0.f:-0.f);b=a;const auto initial=a;auto pc=pp,vc=vv;
  old_acc(a.data()+guard,vv.data(),pp.data());assert(mstar_pv_exact_accumulate64(b.data()+guard,vv.data(),pp.data(),true));
  for(int row=0;row<64;++row)for(int col=0;col<64;++col){const int i=guard+row*64+col;double reference=initial[i],abs_sum=std::abs(reference);
   for(int block=0;block<2;++block){int dot=0;for(int k=block*32;k<(block+1)*32;++k)dot+=quant(pp.data(),row,k)*quant(vv.data(),col,k);
    const double term=double(scale(pp.data(),row,block))*scale(vv.data(),col,block)*dot;reference+=term;abs_sum+=std::abs(term);}
   const double u=0x1p-24,gamma=8*u/(1-8*u);assert(std::isfinite(a[i])&&std::isfinite(b[i]));assert(std::memcmp(&a[i],&b[i],4)==0);
   assert(std::abs(double(a[i])-reference)<=gamma*abs_sum+1e-12);assert(std::abs(double(b[i])-reference)<=gamma*abs_sum+1e-12);
   max_error=std::max(max_error,std::abs(double(b[i])-reference));max_delta=std::max(max_delta,std::abs(double(a[i])-b[i]));changed+=memcmp(&a[i],&b[i],4)!=0;++checked;}
  assert(memcmp(pp.data(),pc.data(),sizeof(pp))==0&&memcmp(vv.data(),vc.data(),sizeof(vv))==0);
  for(size_t i=0;i<guard;++i)assert(a[i]==12345.25f&&b[i]==12345.25f&&a[guard+4096+i]==12345.25f&&b[guard+4096+i]==12345.25f);
 }
 // A false real-domain guard returns before any output write. The unchanged
 // original product+acc route then retains exceptional and signed-zero bytes.
 for(uint32_t bits:{0x7f800000u,0xff800000u,0x7fc12345u,0xffc12345u,0x7f812345u,0xff812345u,0x7f7fffffu,0xff7fffffu}){
  std::array<block_q8_0x4,32> pp{},vv{};for(auto *q:{&pp,&vv})for(auto &block:*q){for(auto &d:block.d)d=ggml_fp32_to_fp16(1.f);for(auto &x:block.qs)x=1;}
  std::array<float,4096> a{},b;memcpy(&a[0],&bits,4);b=a;const auto before=b;
  bool normal=true;for(float x:b)normal=normal&&std::isfinite(x)&&std::abs(x)<=1e30f;assert(!normal);
  assert(!mstar_pv_exact_accumulate64(b.data(),vv.data(),pp.data(),normal));assert(memcmp(b.data(),before.data(),sizeof(b))==0);old_acc(a.data(),vv.data(),pp.data());old_acc(b.data(),vv.data(),pp.data());assert(memcmp(a.data(),b.data(),sizeof(a))==0);++fallback;
 }
#if defined(__aarch64__)
 {uint64_t saved;asm volatile("mrs %0, fpcr":"=r"(saved));
  std::array<float,4096> out{};std::array<block_q8_0x4,32> pp{},vv{};
  for(uint64_t bit:{uint64_t(1)<<22,uint64_t(1)<<23,uint64_t(1)<<24,uint64_t(1)<<25}){
   const uint64_t mode=saved|bit;asm volatile("msr fpcr, %0"::"r"(mode));const bool used=mstar_pv_exact_accumulate64(out.data(),vv.data(),pp.data(),true);asm volatile("msr fpcr, %0"::"r"(saved));assert(!used);for(float x:out)assert(x==0);++fallback;}
 }
#endif
 assert(changed==0);printf("{\"status\":\"EXACT_ACCUMULATOR_CONTROLS_PASS\",\"quantized_operand_FP64_analytic_bound_checks\":%llu,\"changed_FP32_elements\":%llu,\"exceptional_falseguard_unchanged_route_checks\":%llu,\"max_candidate_FP64_error\":%.17g,\"max_old_new_difference\":%.17g,\"canaries_and_inputs_unchanged\":true,\"model_qualification\":false}\n",(unsigned long long)checked,(unsigned long long)changed,(unsigned long long)fallback,max_error,max_delta);
}
}
#ifndef MSTAR_PV_ACCUMULATOR_INCLUDED
int main(){fma_controls();}
#endif
