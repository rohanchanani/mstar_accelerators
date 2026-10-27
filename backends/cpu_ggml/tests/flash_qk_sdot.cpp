// Mechanism only: actual operator vs FP64 attention on independently decoded
// packed operands. This does not substitute for central model qualification.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu/repack.h"
#include "flash_qk_sdot.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
struct Case {int q,k,h,hkv,b,bkv,threads;bool mask,ref,f32;};
static std::vector<double> quantized(const std::vector<float> & rows,int count) {
    const int padded=GGML_PAD(count,4);std::vector<float> input(padded*64,0);std::copy(rows.begin(),rows.end(),input.begin());
    std::vector<block_q8_0x4> packed(padded/4*2);
    for(int row=0;row<padded;row+=4)ggml_quantize_mat_q8_0_4x4(input.data()+row*64,packed.data()+row/4*2,64);
    std::vector<double> result(count*64);
    for(int r=0;r<count;++r)for(int d=0;d<64;++d) {
        const auto & block=packed[r/4*2+d/32];
        result[r*64+d]=double(ggml_fp16_to_fp32(block.d[r%4]))*block.qs[(d%32)/4*16+(r%4)*4+d%4];
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
      {65,128,2,2,1,1,4,0,0,1}};
    int executions=0,quant_checks=0,fallback_checks=0;double maximum=0,float_difference=0;
    for(const auto c:cases) {
        auto * ctx=ggml_init({2u<<20,nullptr,true});assert(ctx);auto * backend=ggml_backend_cpu_init();assert(backend);
        auto * graph=ggml_new_graph(ctx);auto * qr=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,64,c.h,c.q,c.b);
        auto * kr=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,64,c.hkv,c.k,c.bkv);auto * vr=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,64,c.hkv,c.k,c.bkv);
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
        const bool eligible=c.q>=64 && c.k%64==0 && !c.mask && !c.ref && !c.f32;
        for(int repeat=0;repeat<3;++repeat) {
            std::vector<float> qv(ggml_nelements(qr));for(size_t i=0;i<qv.size();++i)qv[i]=float(int((i*7+repeat*13)%47)-23)/31;
            std::vector<ggml_fp16_t> kv(ggml_nelements(kr)),vv(kv.size());for(size_t i=0;i<kv.size();++i){kv[i]=ggml_fp32_to_fp16(float(int((i*11+repeat*17)%53)-26)/29);vv[i]=ggml_fp32_to_fp16(float(int((i*13+repeat*19)%59)-29)/37);}
            ggml_backend_tensor_set(qr,qv.data(),0,ggml_nbytes(qr));ggml_backend_tensor_set(kr,kv.data(),0,ggml_nbytes(kr));ggml_backend_tensor_set(vr,vv.data(),0,ggml_nbytes(vr));
            if(mask){std::vector<ggml_fp16_t> m(ggml_nelements(mask),ggml_fp32_to_fp16(0));for(int row=0;row<c.q;++row)for(int col=c.k-17;col<c.k;++col)m[row*c.k+col]=ggml_fp32_to_fp16(-INFINITY);ggml_backend_tensor_set(mask,m.data(),0,ggml_nbytes(mask));}
            std::vector<float> plain(ggml_nelements(out)),result(plain.size());
            for(int marked=0;marked<2;++marked){tag=marked?MSTAR_ENCODER_QK_SDOT_TAG:0;std::memcpy((char *)out->op_params+16,&tag,4);
                assert(ggml_graph_compute(graph,&plan)==GGML_STATUS_SUCCESS);++executions;
                for(size_t i=0;i<64;++i)assert(work[i]==0xa5 && work[64+plan.work_size+i]==0xa5);
                ggml_backend_tensor_get(out,(marked?result:plain).data(),0,ggml_nbytes(out));}
            if(!eligible){assert(std::memcmp(plain.data(),result.data(),result.size()*4)==0);++fallback_checks;}
            else {
                for(int batch=0;batch<c.b;++batch)for(int head=0;head<c.h;++head) {
                    const int kh=head/(c.h/c.hkv),kb=batch/(c.b/c.bkv);
                    std::vector<float> qs(c.q*64),ks(c.k*64);
                    for(int row=0;row<c.q;++row)std::copy_n(qv.data()+((batch*c.q+row)*c.h+head)*64,64,qs.data()+row*64);
                    for(int row=0;row<c.k;++row)for(int d=0;d<64;++d)ks[row*64+d]=ggml_fp16_to_fp32(kv[((kb*c.k+row)*c.hkv+kh)*64+d]);
                    const auto qquant=quantized(qs,c.q),kquant=quantized(ks,c.k);
                    std::vector<double> scores(c.k),sum(64);
                    for(int row=0;row<c.q;++row){double max=-INFINITY,denom=0;std::fill(sum.begin(),sum.end(),0);
                        for(int key=0;key<c.k;++key){double dot=0;for(int d=0;d<64;++d)dot+=qquant[row*64+d]*kquant[key*64+d];scores[key]=dot*.125;max=std::max(max,scores[key]);}
                        for(int key=0;key<c.k;++key){const double p=std::exp(scores[key]-max);denom+=p;for(int d=0;d<64;++d)sum[d]+=p*ggml_fp16_to_fp32(vv[((kb*c.k+key)*c.hkv+kh)*64+d]);}
                        for(int d=0;d<64;++d){size_t i=((batch*c.q+row)*c.h+head)*64+d;double err=std::abs(result[i]-sum[d]/denom);maximum=std::max(maximum,err);assert(std::isfinite(result[i]) && err<3e-5);float_difference=std::max(float_difference,double(std::abs(result[i]-plain[i])));}
                    }
                }++quant_checks;
            }
            std::vector<ggml_fp16_t> check(kv.size());ggml_backend_tensor_get(kr,check.data(),0,ggml_nbytes(kr));assert(check==kv);ggml_backend_tensor_get(vr,check.data(),0,ggml_nbytes(vr));assert(check==vv);
        }
        std::printf("{\"q\":%d,\"k\":%d,\"heads\":%d,\"kv_heads\":%d,\"batches\":%d,\"threads\":%d,\"eligible\":%s,\"workspace\":%zu}\n",c.q,c.k,c.h,c.hkv,c.b,c.threads,eligible?"true":"false",plan.work_size);
        ggml_gallocr_free(allocator);ggml_free(ctx);ggml_backend_free(backend);
    }
    assert(float_difference>1e-6);
    std::printf("{\"status\":\"PASS\",\"executions\":%d,\"quantized_attention_checks\":%d,\"fallback_byte_checks\":%d,\"max_quantized_reference_error\":%.12g,\"max_original_float_difference\":%.12g}\n",executions,quant_checks,fallback_checks,maximum,float_difference);
}
