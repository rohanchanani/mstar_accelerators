// Bounded physical mechanism/complete PV operator, not model qualification.
#define main retained_pv_probe_main
#include "flash_pv_gemm.cpp"
#undef main
#include "flash_pquant_vector_store.h"
#include <limits>
extern "C" __attribute__((noinline)) void mstar_pquant_vector_probe(const float *x,void *y){mstar_pquant_vector_store64(x,y);}
namespace {
void vector_controls(){
 uint64_t bytes=0,cases=0;uint32_t rng=0x12345678;
 for(int kind=0;kind<12;++kind)for(int repeat=0;repeat<32;++repeat){
  alignas(64) std::array<float,256> x{};
  const uint32_t special[]={0,0x80000000u,1,0x80000001u,0x007fffffu,0x807fffffu,0x00800000u,0x7f7fffffu,0xff7fffffu,0x7f800000u,0xff800000u,0x7fc12345u,0xffc12345u,0x7f812345u,0xff812345u,0x3f800000u};
  for(int i=0;i<256;++i){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;
   if(kind==0)x[i]=0.f;else if(kind==1)x[i]=1.f;else if(kind==2)x[i]=std::exp(-float((i*13+repeat*11)%101)/9.f);
   else if(kind==3)x[i]=float(int(rng%20001)-10000)/19.f;
   else if(kind==4)memcpy(&x[i],&rng,4);
   else {auto bits=special[(i*7+repeat+kind)%16];memcpy(&x[i],&bits,4);}
  }
  auto original=x;alignas(64) std::array<uint8_t,272+128> old,now;old.fill(0xa5);now.fill(0xa5);
  ggml_quantize_mat_q8_0_4x4(x.data(),old.data()+64,64);mstar_pquant_vector_probe(x.data(),now.data()+64);
  if(memcmp(old.data(),now.data(),old.size())){fprintf(stderr,"packed byte mismatch kind%d repeat%d\n",kind,repeat);assert(false);}
  assert(memcmp(x.data(),original.data(),sizeof(x))==0);
  for(int i=0;i<64;++i)assert(old[i]==0xa5&&old[336+i]==0xa5&&now[i]==0xa5&&now[336+i]==0xa5);
  bytes+=272;++cases;
 }
 printf("{\"kind\":\"controls\",\"status\":\"PQUANT_VECTOR_STORE_BYTES_PASS\",\"cases\":%llu,\"packed_bytes_exact\":%llu,\"input_and_canaries_immutable\":true,\"special_values_include_signaling_nan\":true,\"model_qualification\":false}\n",(unsigned long long)cases,(unsigned long long)bytes);
}
void quant_rows(Work&w,bool candidate){for(int row=0;row<64;row+=4){auto *dest=reinterpret_cast<char*>(w.pq.data())+row*68;if(candidate)mstar_pquant_vector_probe(w.p.data()+row*64,dest);else ggml_quantize_mat_q8_0_4x4(w.p.data()+row*64,dest,64);}}
void quant_run(int threads,bool candidate,int repeat){
 std::vector<Work> work(threads);for(auto &w:work)prepare(w,2+repeat%2);int actual=1;std::vector<uint64_t> cpu(threads);auto start=clock_ns(),ps=clock_ns(CLOCK_PROCESS_CPUTIME_ID);
#if defined(_OPENMP)
#pragma omp parallel num_threads(threads)
#endif
 {int ith=0;
#if defined(_OPENMP)
 ith=omp_get_thread_num();
#pragma omp single
 actual=omp_get_num_threads();
#endif
 auto &w=work[ith];auto begin=clock_ns(CLOCK_THREAD_CPUTIME_ID);
 for(int tile=ith;tile<3456;tile+=threads){quant_rows(w,candidate);asm volatile(""::"m"(w.pq):"memory");}
 cpu[ith]=clock_ns(CLOCK_THREAD_CPUTIME_ID)-begin;
 }
 auto wall=clock_ns()-start,pcpu=clock_ns(CLOCK_PROCESS_CPUTIME_ID)-ps;uint64_t team=0,checksum=0;for(auto t:cpu)team+=t;for(auto &w:work)for(auto &b:w.pq)for(auto q:b.qs)checksum+=uint8_t(q);assert(actual==threads);
 printf("{\"kind\":\"Ppack_timing\",\"variant\":\"%s\",\"repeat\":%d,\"actual_threads\":%d,\"PV_equivalent_tiles\":3456,\"quantizer_calls\":55296,\"wall_ns\":%llu,\"team_CPU_ns\":%llu,\"process_CPU_ns\":%llu,\"checksum\":%llu}\n",candidate?"vector_store_exact":"original_scalar_stores",repeat,actual,(unsigned long long)wall,(unsigned long long)team,(unsigned long long)pcpu,(unsigned long long)checksum);
}
std::vector<float> full_run(int threads,bool candidate,int repeat){
    constexpr int H=6,T=1536,Q=1500,NT=24;
    std::vector<Work> work(threads);for(auto &w:work)prepare(w,2+repeat%2);
    
    std::vector<ggml_fp16_t> source(T*384);
    std::vector<float> vf(H*NT*4096),out(H*Q*64);
    std::vector<block_q8_0x4> vq(H*NT*PACKED_BLOCKS);
    for(int k=0;k<T;++k)for(int c=0;c<384;++c)source[k*384+c]=ggml_fp32_to_fp16(k<Q?float((k*17+c*13+repeat*11)%97-48)/31.f:0.f);
    int actual=1;std::vector<uint64_t> cpu(threads);auto start=clock_ns(),ps=clock_ns(CLOCK_PROCESS_CPUTIME_ID);
#if defined(_OPENMP)
#pragma omp parallel num_threads(threads)
#endif
    {int ith=0;
#if defined(_OPENMP)
        ith=omp_get_thread_num();
#pragma omp single
        actual=omp_get_num_threads();
#endif
        auto &w=work[ith];auto ts=clock_ns(CLOCK_THREAD_CPUTIME_ID);
        // Same owner, one shared pack readiness barrier before any consumers.
        for(int tile=ith;tile<H*NT;tile+=threads){int head=tile/NT,key=tile%NT;
            auto *src=source.data()+key*64*384+head*64;
            {auto *v=vf.data()+tile*4096;
                for(int k=0;k<64;++k)ggml_cpu_fp16_to_fp32(src+k*384,v+k*64,64);
                bool finite=true;for(int i=0;i<4096;++i)finite=finite&&std::isfinite(v[i]);assert(finite);
                for(int c=0;c<64;c+=4){for(int r=0;r<4;++r)for(int k=0;k<64;++k)w.rows[r*64+k]=v[k*64+c+r];
                    ggml_quantize_mat_q8_0_4x4(w.rows.data(),reinterpret_cast<char*>(vq.data()+tile*PACKED_BLOCKS)+c*68,64);}
            }
        }
#if defined(_OPENMP)
#pragma omp barrier
#endif
        for(int task=ith;task<H*NT;task+=threads){int head=task/NT,query=task%NT;w.c.fill(0);
            for(int key=0;key<NT;++key){int tile=head*NT+key;
                quant_rows(w,candidate);mstar_pv_gemm4x8(64,w.product.data(),64,vq.data()+tile*PACKED_BLOCKS,w.pq.data(),64,64);
                ggml_vec_acc_f32(4096,w.c.data(),w.product.data());
            }
            const int live=std::min(64,Q-query*64);
            std::memcpy(out.data()+head*Q*64+query*4096,w.c.data(),live*64*sizeof(float));
        }
        cpu[ith]=clock_ns(CLOCK_THREAD_CPUTIME_ID)-ts;
    }
    auto wall=clock_ns()-start,pcpu=clock_ns(CLOCK_PROCESS_CPUTIME_ID)-ps;uint64_t teamcpu=0;double sum=0;for(auto t:cpu)teamcpu+=t;for(float x:out){assert(std::isfinite(x));sum+=x;}
    assert(actual==threads);
    std::printf("{\"kind\":\"full_attention_PV_timing\",\"variant\":\"%s\",\"repeat\":%d,\"requested_threads\":%d,\"actual_threads\":%d,\"heads\":6,\"queries\":1500,\"keys\":1536,\"zero_tail_rows\":36,\"PV_tiles\":3456,\"wall_ns\":%llu,\"team_CPU_ns\":%llu,\"process_CPU_ns\":%llu,\"checksum\":%.12g,\"scope\":\"complete_original_V_gather_guard_pack_once;exact_Ppack_original_or_vectorstore_each_tile;GEMM;F32Cadd;valid_outputcopy;oneOMPteam_existing_readiness_barrier;noQK_or_softmax\"}\n",candidate?"vector_store_exact_complete":"current_q8_4x8_complete",repeat,threads,actual,(unsigned long long)wall,(unsigned long long)teamcpu,(unsigned long long)pcpu,sum);
    return out;
}

}
int main(){
#if !defined(__aarch64__) || !defined(__ARM_NEON)
 return 2;
#else
 uint64_t fpcr;asm volatile("mrs %0, fpcr":"=r"(fpcr));int dot=-1;
#if defined(__linux__) && defined(HWCAP_ASIMDDP)
 dot=bool(getauxval(AT_HWCAP)&HWCAP_ASIMDDP);
#endif
 printf("{\"kind\":\"identity\",\"compiler\":\"%s\",\"ASIMDDP\":%d,\"FPCR\":%llu,\"block_width\":32}\n",__VERSION__,dot,(unsigned long long)fpcr);
 vector_controls();
 for(int threads:{1,2,4}){
#if !defined(_OPENMP)
 if(threads!=1)continue;
#endif
 for(int repeat=0;repeat<3;++repeat){std::array<std::vector<float>,2> outputs;for(bool candidate:repeat%2?std::array<bool,2>{true,false}:std::array<bool,2>{false,true}){quant_run(threads,candidate,repeat);outputs[candidate]=full_run(threads,candidate,repeat);}assert(outputs[0].size()==576000&&outputs[0].size()==outputs[1].size()&&memcmp(outputs[0].data(),outputs[1].data(),outputs[0].size()*4)==0);printf("{\"kind\":\"complete_output_exact\",\"threads\":%d,\"repeat\":%d,\"elements\":576000}\n",threads,repeat);}
 }
 puts("{\"kind\":\"completion\",\"status\":\"PQUANT_VECTOR_STORE_SCREEN_COMPLETE\",\"model_qualification\":false}");return 0;
#endif
}
