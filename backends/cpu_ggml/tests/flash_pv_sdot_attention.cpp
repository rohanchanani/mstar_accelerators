// Mechanism only: exact scalar scaled-integer PV integration, unchanged QK/softmax.
// Original-path byte checks for unsupported/nonfinite inputs; central gate separate.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu/repack.h"
#include "flash_qk_sdot.h"
#include "flash_pv_sdot.h"
#include "flash_softmax_tile.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
struct Case {int q,k,h,hkv,b,bkv,threads;bool mask,ref,f32;int vh=0;bool nonfinite=false;bool badq=false;};

static float add_scaled(float sum,float scale,int dot) {
#if MSTAR_PV_Q8_GEMM_FUSED
    return std::fma(scale,float(dot),sum);
#else
    volatile float product=scale*float(dot);return sum+product;
#endif
}
// Independent scalar PV integer-dot/scale/add reference. QK and unchanged
// softmax reuse their already-qualified primitives; this tests PV integration.
static std::vector<float> reference(const Case &c,const std::vector<float>&qv,
    const std::vector<ggml_fp16_t>&kv,const std::vector<ggml_fp16_t>&vv) {
    std::vector<float> result(size_t(c.b)*c.q*c.h*64);
    for(int b=0;b<c.b;++b)for(int h=0;h<c.h;++h)for(int first=0;first<c.q;first+=64){
        float q[4096]={},k[4096],v[4096],scores[4096],acc[4096]={},sums[64]={},maxima[64];
        std::fill_n(maxima,64,-INFINITY);
        block_q8_0x4 qq[32],kq[32],vq[32],pq[32];
        const int hb=h/(c.h/c.hkv),bb=b/(c.b/c.bkv),vh=c.vh?c.vh:c.hkv,vhindex=h/(c.h/vh);
        for(int row=0;row<std::min(64,c.q-first);++row)
            std::copy_n(qv.data()+((b*c.q+first+row)*c.h+h)*64,64,q+row*64);
        for(int row=0;row<64;row+=4)ggml_quantize_mat_q8_0_4x4(q+row*64,qq+row/4*2,64);
        for(int firstkey=0;firstkey<c.k;firstkey+=64){
            for(int key=0;key<64;++key)for(int d=0;d<64;++d){
                const size_t i=((bb*c.k+firstkey+key)*c.hkv+hb)*64+d;
                k[key*64+d]=ggml_fp16_to_fp32(kv[i]);v[d*64+key]=ggml_fp16_to_fp32(vv[((bb*c.k+firstkey+key)*vh+vhindex)*64+d]);
            }
            for(int row=0;row<64;row+=4){ggml_quantize_mat_q8_0_4x4(k+row*64,kq+row/4*2,64);ggml_quantize_mat_q8_0_4x4(v+row*64,vq+row/4*2,64);}
            ggml_gemm_q8_0_4x4_q8_0(64,scores,64,kq,qq,64,64);ggml_vec_scale_f32(4096,scores,.125f);
            bool skip[64]={};mstar_flash_softmax64(scores,acc,sums,maxima,skip);
            for(int row=0;row<64;row+=4)ggml_quantize_mat_q8_0_4x4(scores+row*64,pq+row/4*2,64);
            auto quant=[](const block_q8_0x4 *p,int row,int col){return int(p[row/4*2+col/32].qs[(col%32)/4*16+(row%4)*4+col%4]);};
            for(int row=0;row<64;++row)for(int ch=0;ch<64;++ch){
                float product=0;
                for(int block=0;block<2;++block){int dot=0;for(int key=block*32;key<(block+1)*32;++key)dot+=quant(pq,row,key)*quant(vq,ch,key);
                    const float scale=ggml_fp16_to_fp32(pq[row/4*2+block].d[row%4])*ggml_fp16_to_fp32(vq[ch/4*2+block].d[ch%4]);
                    product=add_scaled(product,scale,dot);
                }
                acc[row*64+ch]+=product;
            }
        }
        for(int row=0;row<std::min(64,c.q-first);++row){ggml_vec_scale_f32(64,acc+row*64,sums[row]==0?0:1.f/sums[row]);
            std::copy_n(acc+row*64,64,result.data()+((b*c.q+first+row)*c.h+h)*64);}
    }
    return result;
}
int main() {
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_DOTPROD)
    std::fprintf(stderr,"actual ARM DOTPROD control required\n");return 2;
#endif
    const Case cases[]={{65,128,2,2,1,1,1,0,0,0},{65,128,2,2,1,1,2,0,0,0},
      {65,128,2,2,1,1,4,0,0,0},{129,192,4,2,2,2,4,0,0,0},
      {1500,64,6,6,1,1,4,0,0,0},{65,1536,6,6,1,1,4,0,0,0},
      {65,127,2,2,1,1,4,0,0,0},{65,128,2,2,1,1,4,1,0,0},
      {1,128,2,2,1,1,4,0,0,0},{65,128,2,2,1,1,4,0,1,0},
      {65,128,2,2,1,1,4,0,0,1},
      {65,64,16,1,1,1,4,0,0,0,16},
      {65,128,2,2,1,1,4,0,0,0,0,true},
      {65,128,4,2,2,2,4,0,0,0,1},
      {65,128,2,2,1,1,4,0,0,0,0,false,true}};
    int executions=0,bitwise_checks=0,fallback_checks=0;double max_difference=0;
    for(const auto c:cases) {
        auto * ctx=ggml_init({2u<<20,nullptr,true});assert(ctx);auto * backend=ggml_backend_cpu_init();assert(backend);
        auto * graph=ggml_new_graph(ctx);auto * qr=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,64,c.h,c.q,c.b);
        auto * kr=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,64,c.hkv,c.k,c.bkv);auto * vr=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,64,c.vh?c.vh:c.hkv,c.k,c.bkv);
        for(auto * t:{qr,kr,vr})ggml_set_input(t);
        auto * q=ggml_permute(ctx,qr,0,2,1,3);auto * k=ggml_permute(ctx,kr,0,2,1,3);auto * v=ggml_permute(ctx,vr,0,2,1,3);
        ggml_tensor * mask=nullptr;if(c.mask){mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,c.k,GGML_PAD(c.q,64));ggml_set_input(mask);}
        auto * out=ggml_flash_attn_ext(ctx,q,k,v,mask,.125f,0,0);if(c.f32)ggml_prec_set_acc(out,GGML_PREC_F32);
        ggml_set_output(out);ggml_build_forward_expand(graph,out);
        // Reserved words really begin initialized; never reinterpret random bits.
        int32_t tag;std::memcpy(&tag,(char *)out->op_params+16,4);assert(tag==0);
        auto * allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));assert(ggml_gallocr_alloc_graph(allocator,graph));
        auto plan=ggml_graph_plan(graph,c.threads,nullptr);plan.use_ref=c.ref;
        std::vector<uint8_t> work(plan.work_size+128,0xa5);plan.work_data=work.data()+64;
        const bool eligible=c.q>=64 && c.k%64==0 && !c.mask && !c.ref && !c.f32 && (c.hkv+(c.vh?c.vh:c.hkv))*68<=c.hkv*256 && !c.nonfinite && !c.badq;
        for(int repeat=0;repeat<3;++repeat) {
            std::vector<float> qv(ggml_nelements(qr));for(size_t i=0;i<qv.size();++i)qv[i]=float(int((i*7+repeat*13)%47)-23)/31;
            if(c.badq)for(auto &x:qv)x=1e30f;
            std::vector<ggml_fp16_t> kv(ggml_nelements(kr)),vv(ggml_nelements(vr));for(size_t i=0;i<kv.size();++i){kv[i]=ggml_fp32_to_fp16(float(int((i*11+repeat*17)%53)-26)/29);}
            for(size_t i=0;i<vv.size();++i)vv[i]=ggml_fp32_to_fp16(float(int((i*13+repeat*19)%59)-29)/37);
            if(c.nonfinite)for(auto &x:vv)x=ggml_fp32_to_fp16(repeat==0?NAN:repeat==1?INFINITY:-INFINITY);
            ggml_backend_tensor_set(qr,qv.data(),0,ggml_nbytes(qr));ggml_backend_tensor_set(kr,kv.data(),0,ggml_nbytes(kr));ggml_backend_tensor_set(vr,vv.data(),0,ggml_nbytes(vr));
            if(mask){std::vector<ggml_fp16_t> m(ggml_nelements(mask),ggml_fp32_to_fp16(0));for(int row=0;row<c.q;++row)for(int col=c.k-17;col<c.k;++col)m[row*c.k+col]=ggml_fp32_to_fp16(-INFINITY);ggml_backend_tensor_set(mask,m.data(),0,ggml_nbytes(mask));}
            std::vector<float> plain(ggml_nelements(out)),result(plain.size());
            for(int marked=0;marked<2;++marked){tag=MSTAR_ENCODER_QK_SDOT_TAG;std::memcpy((char *)out->op_params+16,&tag,4);
                tag=marked?MSTAR_ENCODER_PV_SDOT_TAG:0;std::memcpy((char *)out->op_params+20,&tag,4);
                assert(ggml_graph_compute(graph,&plan)==GGML_STATUS_SUCCESS);++executions;
                for(size_t i=0;i<64;++i)assert(work[i]==0xa5 && work[64+plan.work_size+i]==0xa5);
                ggml_backend_tensor_get(out,(marked?result:plain).data(),0,ggml_nbytes(out));}
            if(!eligible){assert(std::memcmp(plain.data(),result.data(),result.size()*4)==0);++fallback_checks;}
            else {auto expected=reference(c,qv,kv,vv);assert(expected.size()==result.size());
                if(std::memcmp(expected.data(),result.data(),result.size()*4)!=0){
                    for(size_t i=0;i<result.size();++i)if(std::memcmp(&expected[i],&result[i],4)!=0){std::fprintf(stderr,"PV integration mismatch q=%d k=%d threads=%d index=%zu expected=%a actual=%a\n",c.q,c.k,c.threads,i,double(expected[i]),double(result[i]));break;}assert(false);}
                ++bitwise_checks;
                for(size_t i=0;i<result.size();++i){assert(std::isfinite(result[i]));max_difference=std::max(max_difference,double(std::abs(result[i]-plain[i])));}
            }
            std::vector<ggml_fp16_t> check(kv.size());ggml_backend_tensor_get(kr,check.data(),0,ggml_nbytes(kr));assert(check==kv);check.resize(vv.size());ggml_backend_tensor_get(vr,check.data(),0,ggml_nbytes(vr));assert(std::memcmp(check.data(),vv.data(),vv.size()*2)==0);
        }
        std::printf("{\"q\":%d,\"k\":%d,\"heads\":%d,\"kv_heads\":%d,\"batches\":%d,\"threads\":%d,\"eligible\":%s,\"workspace\":%zu}\n",c.q,c.k,c.h,c.hkv,c.b,c.threads,eligible?"true":"false",plan.work_size);
        ggml_gallocr_free(allocator);ggml_free(ctx);ggml_backend_free(backend);
    }
    assert(max_difference>0);
    std::printf("{\"max_F32_difference\":%.12g,\"model_qualification\":false}\n",max_difference);
    std::printf("{\"status\":\"PV_QUANTIZED_INTEGRATION_PASS\",\"executions\":%d,\"exact_scalar_PV_checks\":%d,\"fallback_byte_checks\":%d}\n",executions,bitwise_checks,fallback_checks);
}
