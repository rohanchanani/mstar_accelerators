// Bounded physical operator probe, not model qualification or a serving path.
#include "common.h"
#include "repack.h"
#include "vec.h"
#include MSTAR_PV_SIMD_HEADER

#include <limits>
extern "C" void baseline_softmax(float*,float*,float*,float*,bool*);
extern "C" void guarded_softmax(float*,float*,float*,float*,bool*);
extern "C" void xnn_softmax(float*,float*,float*,float*,bool*);
extern "C" void cleanup_softmax(float*,float*,float*,float*,bool*);
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
    std::array<float,D*D> p{},v{},c{},product{},scores{};
    std::array<float,64> sums{},maxima{};
    std::array<float,128> pmax{};
    std::array<bool,64> skip{};
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
    w.maxima.fill(-INFINITY);w.sums.fill(0);
    for(size_t i=0;i<w.scores.size();++i)w.scores[i]=float(int((i*17+kind*13)%89)-44)/31.f;
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
void q8_pv(Work&w) {
    quant_p(w);
    ggml_gemm_q8_0_4x4_q8_0(D,w.product.data(),D,w.vq.data(),w.pq.data(),D,D);
    ggml_vec_acc_f32(D*D,w.c.data(),w.product.data());
}
float decoded_scale(const std::array<block_q8_0x4,PACKED_BLOCKS>&p,int row,int block) {return GGML_CPU_FP16_TO_FP32(p[row/4*2+block].d[row%4]);}
int decoded_q(const std::array<block_q8_0x4,PACKED_BLOCKS>&p,int row,int col) {return p[row/4*2+col/32].qs[(col%32)/4*16+(row%4)*4+col%4];}
void controls() {
    double max_quant_vs_f32=0,squared=0;size_t elements=0;
    for(int kind=0;kind<4;++kind){
        Work w;prepare(w,kind);pack_v(w,true);const auto p=w.p,v=w.v,c=w.c;const auto source=w.source;quant_p(w);
        ggml_gemm_q8_0_4x4_q8_0(D,w.product.data(),D,w.vq.data(),w.pq.data(),D,D);
        for(int row=0;row<D;++row)for(int ch=0;ch<D;++ch){
            float expected=0.f;
            for(int block=0;block<2;++block){int integer_dot=0;for(int key=block*32;key<(block+1)*32;++key)integer_dot+=decoded_q(w.pq,row,key)*decoded_q(w.vq,ch,key);
                const float scale=decoded_scale(w.pq,row,block)*decoded_scale(w.vq,ch,block);
                expected=scaled_add(expected,scale,integer_dot);
            }
            if(std::memcmp(&expected,&w.product[row*D+ch],4)!=0) {
                uint32_t eb,ab;std::memcpy(&eb,&expected,4);std::memcpy(&ab,&w.product[row*D+ch],4);
                std::fprintf(stderr,"packed reference mismatch case=%d row=%d channel=%d expected=%a actual=%a expected_bits=%08x actual_bits=%08x fused=%d\n",kind,row,ch,double(expected),double(w.product[row*D+ch]),eb,ab,MSTAR_PV_Q8_GEMM_FUSED);
                assert(false);
            }
        }
        q8_pv(w);auto exact=c;simd_gemm(exact.data(),p.data(),v.data(),D,D,D);
        for(size_t i=0;i<exact.size();++i){assert(std::isfinite(w.c[i]));const float expected_accum=c[i]+w.product[i];assert(std::memcmp(&expected_accum,&w.c[i],4)==0);const double e=std::abs(double(w.c[i])-exact[i]);max_quant_vs_f32=std::max(max_quant_vs_f32,e);squared+=e*e;++elements;}
        assert(w.p==p && w.v==v && w.source==source);
        for(auto * g:{&w.guard0,&w.guard1,&w.guard2})for(auto x:*g)assert(x==0xa5);
    }
    std::printf("{\"kind\":\"controls\",\"status\":\"PACKED_INTEGER_DOT_AND_BOUNDS_PASS\",\"matrix_cases\":4,\"quantized_reference_elements_exact\":16384,\"max_difference_vs_original_F32\":%.9g,\"rms_difference_vs_original_F32\":%.9g,\"model_qualification\":false}\n",max_quant_vs_f32,std::sqrt(squared/elements));
}

float scalar_xnn(float x) {
    const float bias=0x1.8000FEp23f;
    float n=std::fma(x,0x1.715476p+0f,bias);uint32_t bits;std::memcpy(&bits,&n,4);bits<<=23;float scale;std::memcpy(&scale,&bits,4);
    n-=bias;float t=std::fma(n,-0x1.62E430p-1f,x);
    float p=std::fma(0x1.0F9F9Cp-7f,t,0x1.573A1Ap-5f);
    p=std::fma(p,t,0x1.555A80p-3f);p=std::fma(p,t,0x1.FFFDC6p-2f);p=std::fma(p,t,0x1.FFFFF6p-1f);
    t*=scale;return std::fma(p,t,scale);
}
void softmax_controls() {
    size_t exact_rows=0,normal_rows=0,fallback_rows=0,changed_values=0;double squared=0,maxerr=0,maxrel=0;size_t nerr=0;
    for(int kind=0;kind<96;++kind) {
        Work initial,a,b,c;prepare(initial,kind);initial.p=initial.scores;
        for(int row=0;row<64;++row) {
            const int mode=(row+kind)%14;
            initial.sums[row]=float(row%7)/3.f;
            for(int col=0;col<64;++col) {
                if(mode==1)initial.p[row*64+col]=0;
                if(mode==2)initial.p[row*64+col]=-INFINITY;
                if(mode==3 && col==23)initial.p[row*64+col]=NAN;
                if(mode==4 && col==23)initial.p[row*64+col]=INFINITY;
                if(mode==5 && col>=48)initial.p[row*64+col]=-INFINITY;
                if(mode==8 || mode==9)initial.p[row*64+col]=col==63?1.f:(mode==8?-86.f:-86.0001f);
                if(mode==12)initial.p[row*64+col]=col==63?std::numeric_limits<float>::max():-std::numeric_limits<float>::max();
            }
            if(mode==6)initial.maxima[row]=1000.f;
            if(mode==7)initial.maxima[row]=-1.f;
            if(mode==10)initial.maxima[row]=INFINITY;
            if(mode==11)initial.maxima[row]=NAN;
        }
        a=b=c=initial;
        baseline_softmax(a.p.data(),a.c.data(),a.sums.data(),a.maxima.data(),a.skip.data());
        guarded_softmax(b.p.data(),b.c.data(),b.sums.data(),b.maxima.data(),b.skip.data());
        xnn_softmax(c.p.data(),c.c.data(),c.sums.data(),c.maxima.data(),c.skip.data());
        Work cleaned=initial;
        cleanup_softmax(cleaned.p.data(),cleaned.c.data(),cleaned.sums.data(),cleaned.maxima.data(),cleaned.skip.data());
        assert(std::memcmp(cleaned.p.data(),c.p.data(),sizeof(c.p))==0);
        assert(std::memcmp(cleaned.c.data(),c.c.data(),sizeof(c.c))==0);
        assert(std::memcmp(cleaned.sums.data(),c.sums.data(),sizeof(c.sums))==0);
        assert(std::memcmp(cleaned.maxima.data(),c.maxima.data(),sizeof(c.maxima))==0);
        assert(std::memcmp(cleaned.skip.data(),c.skip.data(),sizeof(c.skip))==0);
        auto same=[](const auto &x,const auto &y){return std::memcmp(x.data(),y.data(),sizeof(x))==0;};
        assert(same(a.p,b.p)&&same(a.c,b.c)&&same(a.sums,b.sums)&&same(a.maxima,b.maxima)&&same(a.skip,b.skip));exact_rows+=64;
        assert(same(a.c,c.c)&&same(a.maxima,c.maxima)&&same(a.skip,c.skip));
        for(int row=0;row<64;++row) {
            float maximum=-INFINITY,absolute=0;bool finite=true;
            for(int col=0;col<64;++col){const float x=initial.p[row*64+col];finite&=std::isfinite(x);maximum=std::max(maximum,x);absolute=std::max(absolute,std::abs(x));}
            const float next=fmaxf(initial.maxima[row],maximum);
            const bool normal=finite && maximum!=0 && std::isfinite(next) && -absolute-next>=-87.f;
            if(!normal){++fallback_rows;assert(std::memcmp(a.p.data()+row*64,c.p.data()+row*64,64*4)==0);assert(std::memcmp(&a.sums[row],&c.sums[row],4)==0);continue;}
            ++normal_rows;float aa[4]={},bb[4]={};
            for(int col=0;col<64;++col){const float input=initial.p[row*64+col]-next;assert(input>=-87 && input<=0);const float expected=scalar_xnn(input);assert(std::memcmp(&expected,&c.p[row*64+col],4)==0);assert(std::isfinite(expected)&&expected>=0);changed_values+=std::memcmp(&a.p[row*64+col],&expected,4)!=0;const double ref=std::exp(double(input)),e=std::abs(double(expected)-ref);squared+=e*e;maxerr=std::max(maxerr,e);maxrel=std::max(maxrel,e/ref);++nerr;}
            for(int i=0;i<64;i+=8)for(int lane=0;lane<4;++lane){aa[lane]+=c.p[row*64+i+lane];bb[lane]+=c.p[row*64+i+4+lane];}
            float z[4];for(int lane=0;lane<4;++lane)z[lane]=aa[lane]+bb[lane];const float sum=(z[0]+z[1])+(z[2]+z[3]);float previous=initial.sums[row];if(next>initial.maxima[row])previous*=expf(initial.maxima[row]-next);const float expected_sum=float(ggml_float(previous)+ggml_float(sum));assert(std::memcmp(&expected_sum,&c.sums[row],4)==0);
        }
    }
    assert(changed_values>0 && normal_rows>0 && fallback_rows>0);
    std::printf("{\"kind\":\"softmax_controls\",\"status\":\"EXACT_GUARDED_GGML_AND_XNN_SCALAR_ARITHMETIC_PASS\",\"cases\":96,\"exact_guarded_rows\":%zu,\"xnn_normal_rows\":%zu,\"exact_fallback_rows\":%zu,\"xnn_changed_P_values\":%zu,\"xnn_max_abs_exp_error_vs_FP64\":%.12g,\"xnn_max_relative_exp_error_vs_FP64\":%.12g,\"xnn_rms_exp_error_vs_FP64\":%.12g,\"model_qualification\":false,\"error_statistics_are_not_a_model_acceptance_policy\":true}\n",exact_rows,normal_rows,fallback_rows,changed_values,maxerr,maxrel,std::sqrt(squared/nerr));
}
void run(int threads,int variant,int repeat) {
    std::vector<Work> work(threads);for(int i=0;i<threads;++i)prepare(work[i],2);
    std::array<std::array<float,4096>,24> tiles;
    for(int j=0;j<24;++j)for(int row=0;row<64;++row)for(int col=0;col<64;++col)tiles[j][row*64+col]=float(int((row*64+col)*17%89)-44)/31.f+(row%3==0?0.f:row%3==1?j*.03f:-j*.03f);
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
            pack_v(w,true);w.c.fill(0);w.sums.fill(0);w.maxima.fill(-INFINITY);
            for(int j=0;j<REUSE;++j) {
                std::memcpy(w.p.data(),tiles[j].data(),sizeof(w.p));w.skip.fill(false);
                if(variant==0)baseline_softmax(w.p.data(),w.c.data(),w.sums.data(),w.maxima.data(),w.skip.data());
                else if(variant==1)guarded_softmax(w.p.data(),w.c.data(),w.sums.data(),w.maxima.data(),w.skip.data());
                else xnn_softmax(w.p.data(),w.c.data(),w.sums.data(),w.maxima.data(),w.skip.data());
                quant_p(w);
                ggml_gemm_q8_0_4x4_q8_0(D,w.product.data(),D,w.vq.data(),w.pq.data(),D,D);
                ggml_vec_acc_f32(D*D,w.c.data(),w.product.data());
            }
            for(int row=0;row<64;++row)for(int col=0;col<64;++col)w.checksum+=w.c[row*64+col]/w.sums[row];
        }
        cpu[ith]=clock_ns(CLOCK_THREAD_CPUTIME_ID)-tcpu;
    }
    const auto end=clock_ns(),pcpu=clock_ns(CLOCK_PROCESS_CPUTIME_ID)-process;double checksum=0;uint64_t teamcpu=0;for(int i=0;i<actual;++i){checksum+=work[i].checksum;teamcpu+=cpu[i];}
    assert(actual==threads && std::isfinite(checksum));
    std::printf("{\"kind\":\"timing\",\"variant\":\"%s\",\"repeat\":%d,\"requested_threads\":%d,\"actual_threads\":%d,\"groups_per_thread\":%d,\"PV_calls_per_V_pack\":%d,\"PV_calls_total\":%d,\"wall_ns\":%llu,\"process_CPU_ns\":%llu,\"summed_worker_CPU_ns\":%llu,\"checksum\":%.12g,\"scope\":\"24precomputedtiles_stationary_increasing_decreasingmax+copy+softmax_guard_onlineupdate+Pquant+SDOT+F32accum_finalnormalization;Vpack1per24\"}\n",variant==0?"current_complete":variant==1?"guarded_ggml_complete":"xnn_complete",repeat,threads,actual,GROUPS,REUSE,actual*GROUPS*REUSE,(unsigned long long)(end-begin),(unsigned long long)pcpu,(unsigned long long)teamcpu,checksum);
}
}
int main(int argc, char ** argv) {
    std::setvbuf(stdout,nullptr,_IOLBF,0);
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_DOTPROD)
    std::fprintf(stderr,"actual AArch64 DOTPROD compile required\n");return 2;
#endif
    std::printf("{\"kind\":\"identity\",\"compiler\":\"%s\",\"Q8_scaled_dot_fused\":%d}\n",__VERSION__,MSTAR_PV_Q8_GEMM_FUSED);
#if defined(__linux__)
    const auto hwcap=getauxval(AT_HWCAP);
    std::printf("{\"kind\":\"hardware\",\"HWCAP_ASIMDHP\":%d,\"HWCAP_ASIMDFHM\":%d,\"HWCAP_ASIMDDP\":%d}\n",bool(hwcap&HWCAP_ASIMDHP),bool(hwcap&HWCAP_ASIMDFHM),bool(hwcap&HWCAP_ASIMDDP));
#endif
    controls();softmax_controls();
    if(argc==2 && std::strcmp(argv[1],"--controls-only")==0) {
        std::puts("{\"kind\":\"completion\",\"status\":\"SOFTMAX_CLEANUP_BYTE_CONTROLS_PASS\"}");
        return 0;
    }
    for(int threads:{1,2,4}) {
#if !defined(_OPENMP)
        if(threads!=1){std::printf("{\"kind\":\"skip\",\"threads\":%d,\"reason\":\"host_has_no_OpenMP\"}\n",threads);continue;}
#endif
        for(int repeat=0;repeat<4;++repeat)for(int variant:(repeat%2?std::array<int,3>{2,1,0}:std::array<int,3>{0,1,2}))run(threads,variant,repeat);
    }
    std::puts("{\"kind\":\"completion\",\"status\":\"SOFTMAX_EXP_OPERATOR_SCREEN_COMPLETE\",\"model_qualification\":false}");
}
