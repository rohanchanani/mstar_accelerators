// Same exponentials/exception paths; finite sum roundoff control, not qualification.
#include "ggml.h"
#include "ggml-cpu/ggml-cpu-impl.h"
#include "flash_softmax_tile.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <limits>
#include <cstdint>
static void original(float * scores,float * values,float * sums,float * maxima,bool * skip){
    for(int row=0;row<64;++row){float * x=scores+row*64;float tile_max;ggml_vec_max_f32(64,&tile_max,x);
        if(tile_max==-INFINITY){skip[row]=true;continue;}
        const float previous=maxima[row],next=fmaxf(previous,tile_max);
        if(next>previous){const float factor=expf(previous-next);ggml_vec_scale_f32(64,values+row*64,factor);sums[row]*=factor;}
        maxima[row]=next;sums[row]+=ggml_vec_soft_max_f32(64,x,x,next);
    }
}
static float bits(uint32_t x){float f;memcpy(&f,&x,4);return f;}
int main(){
#if !defined(__aarch64__) || !defined(__ARM_NEON)
    std::fprintf(stderr,"actual ARM NEON control required\n");return 2;
#endif
    const float specials[]={0.f,-0.f,std::numeric_limits<float>::denorm_min(),-std::numeric_limits<float>::denorm_min(),
        1.f,-1.f,INFINITY,-INFINITY,bits(0x7fc12345),bits(0xffc12345),bits(0x7f812345),bits(0xff812345)};
    double max_sum_abs=0,max_sum_rel=0,max_normalized_error=0;int changed_sum_rows=0;
    uint64_t executions=0;uint32_t state=0x764123ab;int vector_rows=0,fallback_rows=0;
    for(int kind=0;kind<80;++kind){
        std::vector<float> x(4096),values(4096),sums(64),maxima(64);
        for(size_t i=0;i<x.size();++i){state=state*1664525u+1013904223u;x[i]=float(int(state%70001)-35000)/1379;values[i]=float(int((i*17+kind*7)%331)-165)/179;}
        for(int row=0;row<64;++row){maxima[row]=row%3?-INFINITY:float(row%7-3);sums[row]=float(row%11)/17;
            if(kind==1)for(int d=0;d<64;++d)x[row*64+d]=d%2?-0.f:0.f;
            if(kind==2)for(int d=0;d<64;++d)x[row*64+d]=-INFINITY;
            if(kind==3)for(int d=0;d<64;++d)x[row*64+d]=specials[(row+d)%12];
            if(kind>=4 && kind<16)x[row*64+(row*13)%64]=specials[kind-4];
            if(kind==16)for(int d=0;d<64;++d)x[row*64+d]=-float((d%7)+1)*std::numeric_limits<float>::denorm_min();
            if(kind==17)for(int d=0;d<64;++d)x[row*64+d]=float((d%7)+1)*std::numeric_limits<float>::denorm_min();
            if(kind==18){x[row*64]=0;for(int d=1;d<64;++d)x[row*64+d]=-float(d);}
            if(kind==19){x[row*64]=-0.f;for(int d=1;d<64;++d)x[row*64+d]=-float(d);}
            if(kind==20)maxima[row]=specials[row%12];
            if(kind==21)for(int d=0;d<64;++d)values[row*64+d]=specials[(row+d)%12];
            if(kind==22)for(int d=0;d<64;++d)x[row*64+d]=float(d-row)*1.e30f;
            float a,b;bool fast=mstar_flash_finite_max64(x.data()+row*64,&a);ggml_vec_max_f32(64,&b,x.data()+row*64);if(fast){assert(memcmp(&a,&b,4)==0);++vector_rows;}else ++fallback_rows;
        }
        for(int trial=0;trial<3;++trial){
            auto ox=x,ov=values,os=sums,om=maxima;bool old_skip[64]={},new_skip[64]={};auto nx=x,nv=values,ns=sums,nm=maxima;
            original(ox.data(),ov.data(),os.data(),om.data(),old_skip);mstar_flash_softmax64(nx.data(),nv.data(),ns.data(),nm.data(),new_skip);
            assert(memcmp(ox.data(),nx.data(),4096*4)==0 && memcmp(ov.data(),nv.data(),4096*4)==0 && memcmp(om.data(),nm.data(),64*4)==0 && memcmp(old_skip,new_skip,64)==0);
            for(int row=0;row<64;++row){
                float tile_max;const bool fast=mstar_flash_finite_max64(x.data()+row*64,&tile_max);
                if(!fast)assert(memcmp(&os[row],&ns[row],4)==0);
                assert(std::isfinite(os[row])==std::isfinite(ns[row])&&std::isnan(os[row])==std::isnan(ns[row]));
                if(std::isfinite(os[row])){
                    const double d=std::abs(double(os[row])-ns[row]);
                    max_sum_abs=std::max(max_sum_abs,d);
                    max_sum_rel=std::max(max_sum_rel,d/std::max(1.e-30,std::abs(double(os[row]))));
                    changed_sum_rows+=memcmp(&os[row],&ns[row],4)!=0;
                    if(os[row]>0&&ns[row]>0)for(int j=0;j<64;++j)
                        max_normalized_error=std::max(max_normalized_error,std::abs(double(ox[row*64+j])/os[row]-double(nx[row*64+j])/ns[row]));
                }
            }
            ++executions;
        }
    }
    assert(vector_rows>0&&fallback_rows>0&&max_sum_rel<=16*std::numeric_limits<float>::epsilon());
    std::printf("{\"status\":\"PASS\",\"executions\":%llu,\"finite_vector_rows\":%d,\"exceptional_or_zero_fallback_rows\":%d,\"all_except_finite_sum_bits_equal\":true,\"changed_finite_sum_rows\":%d,\"max_sum_abs_error\":%.17g,\"max_sum_relative_error\":%.17g,\"max_normalized_probability_error\":%.17g}\n",(unsigned long long)executions,vector_rows,fallback_rows,changed_sum_rows,max_sum_abs,max_sum_rel,max_normalized_error);
}
