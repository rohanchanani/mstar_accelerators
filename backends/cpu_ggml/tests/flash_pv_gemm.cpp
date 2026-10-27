// Bounded physical operator probe, not model qualification or a serving path.
#include "common.h"
#include "repack.h"
#include "flash_pv_gemm.h"
#include "vec.h"
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#if !defined(MSTAR_PV_Q8_GEMM_FUSED) || (MSTAR_PV_Q8_GEMM_FUSED != 0 && MSTAR_PV_Q8_GEMM_FUSED != 1)
#error "Declare the inspected archive's vmlaq contraction: 0 separate FMUL/FADD, 1 FMLA"
#endif
namespace {
float scaled_add(float sum,float scale,int dot) {
#if MSTAR_PV_Q8_GEMM_FUSED
    return std::fma(scale,float(dot),sum);
#else
    // Match the pinned GCC archive's separate FMUL then FADD; prevent contraction.
    volatile float rounded_product=scale*float(dot);
    return sum+rounded_product;
#endif
}
constexpr int D=64, REUSE=24, GROUPS=32;
constexpr int PACKED_BLOCKS=D/4*2;
struct alignas(64) Work {
    std::array<float,D*D> p{},v{},c{},product{};
    std::array<ggml_fp16_t,D*384> source{};
    std::array<uint8_t,64> guard0{};
    std::array<block_q8_0x4,PACKED_BLOCKS> pq{};
    std::array<uint8_t,64> guard1{};
    std::array<block_q8_0x4,PACKED_BLOCKS> vq{};
    std::array<uint8_t,64> guard2{};
    std::array<float,4*D> rows{};
    double checksum=0;
};
uint64_t clock_ns(clockid_t id=CLOCK_MONOTONIC){timespec t{};assert(clock_gettime(id,&t)==0);return uint64_t(t.tv_sec)*1000000000+t.tv_nsec;}
void prepare(Work &w,int kind) {
    w.guard0.fill(0xa5);w.guard1.fill(0xa5);w.guard2.fill(0xa5);
    for(int row=0;row<D;++row)for(int col=0;col<D;++col) {
        const int i=row*D+col;
        w.p[i]=kind==0?0.f:kind==1?1.f:kind==2?std::exp(-float((i*13+7)%101)/9.f):(col==row?1.f:std::exp(-float((i*11)%127)/7.f));
        w.c[i]=float(int((i*13+kind*7)%53)-26)/17.f;
    }
    for(size_t i=0;i<w.source.size();++i)w.source[i]=ggml_fp32_to_fp16(float(int((i*17+kind*3)%97)-48)/31.f);
}
void pack_v(Work&w,bool quantize) {
    for(int key=0;key<D;++key)ggml_cpu_fp16_to_fp32(w.source.data()+key*384+3*D,w.v.data()+key*D,D);
    if(!quantize)return;
    for(int ch=0;ch<D;ch+=4){
        for(int r=0;r<4;++r)for(int key=0;key<D;++key)w.rows[r*D+key]=w.v[key*D+ch+r];
        ggml_quantize_mat_q8_0_4x4(w.rows.data(),reinterpret_cast<char*>(w.vq.data())+ch*68,D);
    }
}
void quant_p(Work&w) {
    for(int row=0;row<D;row+=4)ggml_quantize_mat_q8_0_4x4(w.p.data()+row*D,reinterpret_cast<char*>(w.pq.data())+row*68,D);
}
void q8_pv(Work&w,bool candidate) {
    quant_p(w);
    if(candidate)mstar_pv_gemm4x8(D,w.product.data(),D,w.vq.data(),w.pq.data(),D,D);
    else ggml_gemm_q8_0_4x4_q8_0(D,w.product.data(),D,w.vq.data(),w.pq.data(),D,D);
    ggml_vec_acc_f32(D*D,w.c.data(),w.product.data());
}
float decoded_scale(const std::array<block_q8_0x4,PACKED_BLOCKS>&p,int row,int block) {return GGML_CPU_FP16_TO_FP32(p[row/4*2+block].d[row%4]);}
int decoded_q(const std::array<block_q8_0x4,PACKED_BLOCKS>&p,int row,int col) {return p[row/4*2+col/32].qs[(col%32)/4*16+(row%4)*4+col%4];}
void controls() {
    uint64_t exact=0,baseline_mismatches=0;constexpr size_t guard=19;
    for(int kind=0;kind<20;++kind) {
        Work w;prepare(w,kind%4);pack_v(w,true);quant_p(w);const auto original_c=w.c;
        if(kind>=4) {
            const float scales[]={0.f,-0.f,1.f,-1.f,65504.f,0x1p-24f,0x1p-14f,0.03125f};
            for(int row=0;row<16;++row)for(int block=0;block<2;++block){
                auto &a=w.pq[row*2+block];auto &v=w.vq[row*2+block];
                for(int j=0;j<4;++j){a.d[j]=ggml_fp32_to_fp16(scales[(kind+row+j)%8]);v.d[j]=ggml_fp32_to_fp16(scales[(kind+block+j+3)%8]);}
                for(int j=0;j<128;++j){a.qs[j]=int8_t((kind+j*17+row*29)%256-128);v.qs[j]=int8_t((kind*5+j*13+row*23)%256-128);}
            }
        }
        const auto pq=w.pq,vq=w.vq;std::array<float,4096> reference{};
        for(int row=0;row<64;++row)for(int col=0;col<64;++col){
            float sum=0;
            for(int block=0;block<2;++block){int dot=0;for(int k=block*32;k<(block+1)*32;++k)dot+=decoded_q(w.pq,row,k)*decoded_q(w.vq,col,k);
                float scale=decoded_scale(w.pq,row,block)*decoded_scale(w.vq,col,block);volatile float product=scale*float(dot);sum=sum+product;}
            reference[row*64+col]=sum;
        }
        std::array<float,4096+2*guard> candidate,baseline;candidate.fill(12345.25f);baseline.fill(12345.25f);
        mstar_pv_gemm4x8(64,candidate.data()+guard,64,w.vq.data(),w.pq.data(),64,64);
        ggml_gemm_q8_0_4x4_q8_0(64,baseline.data()+guard,64,w.vq.data(),w.pq.data(),64,64);
        assert(std::memcmp(candidate.data()+guard,reference.data(),sizeof(reference))==0);
        for(int i=0;i<4096;++i){baseline_mismatches+=std::memcmp(&baseline[guard+i],&reference[i],4)!=0;const float expected=original_c[i]+reference[i];w.product[i]=candidate[guard+i];w.c[i]=original_c[i];ggml_vec_acc_f32(1,&w.c[i],&w.product[i]);assert(std::memcmp(&expected,&w.c[i],4)==0);++exact;}
        assert(std::memcmp(w.pq.data(),pq.data(),sizeof(pq))==0&&std::memcmp(w.vq.data(),vq.data(),sizeof(vq))==0);
        for(size_t i=0;i<guard;++i)assert(candidate[i]==12345.25f&&candidate[guard+4096+i]==12345.25f&&baseline[i]==12345.25f&&baseline[guard+4096+i]==12345.25f);
        for(auto *g:{&w.guard0,&w.guard1,&w.guard2})for(auto x:*g)assert(x==0xa5);
        // Every other legal shape/stride stays on the original implementation.
        for(auto shape:{std::array<int,4>{32,4,4,11},{64,8,8,13},{64,64,64,80},{64,4,12,17},{96,4,4,9}}){
            int n=shape[0],nr=shape[1],nc=shape[2],stride=shape[3];std::vector<block_q8_0x4> aa(nr/4*(n/32)),bb(nc/4*(n/32));
            for(size_t i=0;i<aa.size();++i)std::memcpy(&aa[i],&pq[i%pq.size()],sizeof(aa[i]));for(size_t i=0;i<bb.size();++i)std::memcpy(&bb[i],&vq[i%vq.size()],sizeof(bb[i]));
            std::vector<float> x(nr*stride+guard,4321.5f),y=x;mstar_pv_gemm4x8(n,x.data(),stride,bb.data(),aa.data(),nr,nc);ggml_gemm_q8_0_4x4_q8_0(n,y.data(),stride,bb.data(),aa.data(),nr,nc);assert(std::memcmp(x.data(),y.data(),x.size()*4)==0);
            for(int row=0;row<nr;++row)for(int col=nc;col<stride;++col)assert(x[row*stride+col]==4321.5f);for(size_t i=nr*stride;i<x.size();++i)assert(x[i]==4321.5f);
        }
    }
#if !MSTAR_PV_Q8_GEMM_FUSED
    assert(baseline_mismatches==0);
#endif
    std::printf("{\"kind\":\"controls\",\"status\":\"EXACT_PV_GEMM_AND_FALLBACK_PASS\",\"matrix_cases\":20,\"candidate_scalar_elements_exact\":%llu,\"outside_shape_cases\":100,\"baseline_different_rounding_elements\":%llu,\"nonzero_C_add_exact\":true,\"all_canaries_immutable_operands\":true}\n",(unsigned long long)exact,(unsigned long long)baseline_mismatches);
}
void run(int threads,bool candidate,int repeat) {
    std::vector<Work> work(threads);for(int i=0;i<threads;++i)prepare(work[i],2);
    int actual=1;std::vector<uint64_t> cpu(threads,0);
    const auto begin=clock_ns(),process=clock_ns(CLOCK_PROCESS_CPUTIME_ID);
#if defined(_OPENMP)
#pragma omp parallel num_threads(threads)
#endif
    {
        int ith=0;
#if defined(_OPENMP)
        ith=omp_get_thread_num();
#pragma omp single
        actual=omp_get_num_threads();
#endif
        Work &w=work[ith];const auto tcpu=clock_ns(CLOCK_THREAD_CPUTIME_ID);
        for(int group=0;group<GROUPS;++group){
            pack_v(w,true);
            for(int j=0;j<REUSE;++j) {
                q8_pv(w,candidate);
            }
        }
        for(float x:w.c)w.checksum+=x;
        cpu[ith]=clock_ns(CLOCK_THREAD_CPUTIME_ID)-tcpu;
    }
    const auto end=clock_ns(),pcpu=clock_ns(CLOCK_PROCESS_CPUTIME_ID)-process;double checksum=0;uint64_t teamcpu=0;for(int i=0;i<actual;++i){checksum+=work[i].checksum;teamcpu+=cpu[i];}
    assert(actual==threads && std::isfinite(checksum));
    std::printf("{\"kind\":\"timing\",\"variant\":\"%s\",\"repeat\":%d,\"requested_threads\":%d,\"actual_threads\":%d,\"groups_per_thread\":%d,\"PV_calls_per_V_pack\":%d,\"PV_calls_total\":%d,\"wall_ns\":%llu,\"process_CPU_ns\":%llu,\"summed_worker_CPU_ns\":%llu,\"checksum\":%.12g,\"scope\":\"V_decode_and_optional_transpose_quant_once_per24_PV_calls; every_PV_includes_Pquant_SDOT_FP32accum; one_OpenMP_entry_perbatch\"}\n",candidate?"q8_4x8_complete":"q8_4x4_complete",repeat,threads,actual,GROUPS,REUSE,actual*GROUPS*REUSE,(unsigned long long)(end-begin),(unsigned long long)pcpu,(unsigned long long)teamcpu,checksum);
}
}
#if !defined(MSTAR_PV_SHARED)
int main() {
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    std::printf("{\"kind\":\"arithmetic_contract\",\"Q8_scaled_dot_fused\":%d}\n",MSTAR_PV_Q8_GEMM_FUSED);
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_DOTPROD)
    std::fprintf(stderr,"actual AArch64 DOTPROD compile required\n");return 2;
#endif
    int fhm=-1,half=-1;
#if defined(__linux__) && defined(HWCAP_ASIMDFHM) && defined(HWCAP_ASIMDHP)
    const auto hw=getauxval(AT_HWCAP);fhm=bool(hw&HWCAP_ASIMDFHM);half=bool(hw&HWCAP_ASIMDHP);
#endif
    std::printf("{\"kind\":\"identity\",\"compiler\":\"%s\",\"HWCAP_ASIMDHP\":%d,\"HWCAP_ASIMDFHM\":%d,\"packed_tile_bytes\":%zu}\n",__VERSION__,half,fhm,sizeof(Work::pq));
    controls();
    for(int threads:{1,2,4}) {
#if !defined(_OPENMP)
        if(threads!=1){std::printf("{\"kind\":\"skip\",\"threads\":%d,\"reason\":\"host_has_no_OpenMP\"}\n",threads);continue;}
#endif
        for(int repeat=0;repeat<4;++repeat)for(bool quantized:(repeat%2?std::array<bool,2>{true,false}:std::array<bool,2>{false,true}))run(threads,quantized,repeat);
    }
    std::puts("{\"kind\":\"completion\",\"status\":\"PV_GEMM_SCREEN_COMPLETE\",\"model_qualification\":false}");
}

#endif

#if defined(MSTAR_PV_SHARED)
#include <unistd.h>
#include <sys/syscall.h>
struct Fixture { std::vector<Work> work;std::vector<int> tids;int threads;explicit Fixture(int n):work(n),tids(n),threads(n){for(auto &w:work)prepare(w,2);} };
extern "C" void * pv_probe_create(int threads) {
    if(threads!=1&&threads!=2&&threads!=4)return nullptr;auto *p=new Fixture(threads);
#pragma omp parallel num_threads(threads)
    {assert(omp_get_num_threads()==threads);int i=omp_get_thread_num();p->tids[i]=int(syscall(SYS_gettid));pack_v(p->work[i],true);q8_pv(p->work[i],false);}
    return p;
}
extern "C" int pv_probe_tid(void *handle,int index){auto *p=static_cast<Fixture *>(handle);return index>=0&&index<p->threads?p->tids[index]:-1;}
extern "C" uint64_t pv_probe_run(void *handle,int candidate){auto *p=static_cast<Fixture *>(handle);auto start=clock_ns();
#pragma omp parallel num_threads(p->threads)
    {assert(omp_get_num_threads()==p->threads);int i=omp_get_thread_num();assert(p->tids[i]==int(syscall(SYS_gettid)));for(int group=0;group<GROUPS;++group){pack_v(p->work[i],true);for(int j=0;j<REUSE;++j)q8_pv(p->work[i],candidate!=0);}}
    return clock_ns()-start;
}
extern "C" double pv_probe_checksum(void *handle){double sum=0;for(auto &w:static_cast<Fixture *>(handle)->work)for(float x:w.c){assert(std::isfinite(x));sum+=x;}return sum;}
extern "C" void pv_probe_destroy(void *handle){delete static_cast<Fixture *>(handle);}
#endif
