// Raw pinned-GGML mechanism: compare executable outputs to unchanged archive.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
struct Case {int q,k,h,hkv,b,bkv,threads;bool mask,ref;};
int main(int argc,char ** argv) {
    assert(argc==2);
    const Case cases[]={{65,128,2,2,1,1,1,false,false},{65,128,2,2,1,1,2,false,false},
        {65,128,2,2,1,1,4,false,false},{129,192,4,2,2,2,4,false,false},
        {65,1536,6,6,1,1,4,false,false},{65,127,2,2,1,1,4,false,false},
        {65,128,2,2,1,1,4,true,false},{1,128,2,2,1,1,4,false,false},
        {65,128,2,2,1,1,4,false,true}};
    for(int ci=0;ci<9;++ci) {
        const auto c=cases[ci];auto * ctx=ggml_init({2u<<20,nullptr,true});assert(ctx);
        auto * backend=ggml_backend_cpu_init();assert(backend);
        auto * graph=ggml_new_graph(ctx);
        auto * qr=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,64,c.h,c.q,c.b);
        auto * kr=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,64,c.hkv,c.k,c.bkv);
        auto * vr=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,64,c.hkv,c.k,c.bkv);
        for(auto * t:{qr,kr,vr})ggml_set_input(t);
        auto * q=ggml_permute(ctx,qr,0,2,1,3);auto * k=ggml_permute(ctx,kr,0,2,1,3);auto * v=ggml_permute(ctx,vr,0,2,1,3);
        ggml_tensor * mask=nullptr;
        if(c.mask) {mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,c.k,GGML_PAD(c.q,64));ggml_set_input(mask);}
        auto * out=ggml_flash_attn_ext(ctx,q,k,v,mask,.125f,0,0);ggml_set_output(out);ggml_build_forward_expand(graph,out);
        auto * allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));assert(ggml_gallocr_alloc_graph(allocator,graph));
        auto plan=ggml_graph_plan(graph,c.threads,nullptr);plan.use_ref=c.ref;
        std::vector<uint8_t> work(plan.work_size+128,0xa5);plan.work_data=work.data()+64;
        for(int repeat=0;repeat<3;++repeat) {
            std::vector<float> qv(ggml_nelements(qr));for(size_t i=0;i<qv.size();++i)qv[i]=float(int((i*7+repeat*13)%47)-23)/32;
            std::vector<ggml_fp16_t> kv(ggml_nelements(kr)),vv(kv.size());for(size_t i=0;i<kv.size();++i) {kv[i]=ggml_fp32_to_fp16(float(int((i*11+repeat*17)%53)-26)/32);vv[i]=ggml_fp32_to_fp16(float(int((i*13+repeat*19)%59)-29)/32);}
            ggml_backend_tensor_set(qr,qv.data(),0,ggml_nbytes(qr));ggml_backend_tensor_set(kr,kv.data(),0,ggml_nbytes(kr));ggml_backend_tensor_set(vr,vv.data(),0,ggml_nbytes(vr));
            if(mask) {std::vector<ggml_fp16_t> m(ggml_nelements(mask),ggml_fp32_to_fp16(0));for(int row=0;row<c.q;++row)for(int col=c.k-17;col<c.k;++col)m[row*c.k+col]=ggml_fp32_to_fp16(-INFINITY);ggml_backend_tensor_set(mask,m.data(),0,ggml_nbytes(mask));}
            assert(ggml_graph_compute(graph,&plan)==GGML_STATUS_SUCCESS);
            for(size_t i=0;i<64;++i)assert(work[i]==0xa5 && work[64+plan.work_size+i]==0xa5);
            std::vector<ggml_fp16_t> check(kv.size());ggml_backend_tensor_get(kr,check.data(),0,ggml_nbytes(kr));assert(check==kv);ggml_backend_tensor_get(vr,check.data(),0,ggml_nbytes(vr));assert(check==vv);
            std::vector<float> result(ggml_nelements(out));ggml_backend_tensor_get(out,result.data(),0,ggml_nbytes(out));
            const std::string path=std::string(argv[1])+"/case-"+std::to_string(ci)+"-"+std::to_string(repeat)+".f32";std::ofstream stream(path,std::ios::binary);assert(stream.write((char *)result.data(),result.size()*sizeof(float)));
        }
        std::printf("{\"case\":%d,\"queries\":%d,\"keys\":%d,\"heads\":%d,\"kv_heads\":%d,\"batches\":%d,\"kv_batches\":%d,\"threads\":%d,\"mask\":%d,\"reference_path\":%d,\"work_size\":%zu}\n",ci,c.q,c.k,c.h,c.hkv,c.b,c.bkv,c.threads,c.mask,c.ref,plan.work_size);
        ggml_gallocr_free(allocator);ggml_free(ctx);ggml_backend_free(backend);
    }
}
