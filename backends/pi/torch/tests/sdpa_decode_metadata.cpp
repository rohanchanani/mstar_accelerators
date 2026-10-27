// Host metadata controls only; no native context, model oracle or GPU evidence.
#include "managed-sdpa-decode.h"
#include <cassert>
#include <iostream>

static ggml_tensor tensor(int64_t d, int64_t h, int64_t t, int64_t b) {
    ggml_tensor result{};
    result.type = GGML_TYPE_F32;
    result.ne[0]=d; result.ne[1]=h; result.ne[2]=t; result.ne[3]=b;
    result.nb[0]=4;
    for (int i=1; i<4; ++i) result.nb[i]=result.nb[i-1]*result.ne[i-1];
    return result;
}

int main() {
    int passed = 0;
    managed_sdpa_decode::Parameters p{};
    for (const int t : {1, 17, 448, 1500, 1536}) for (const int b : {1, 2}) {
        auto q=tensor(64,6,1,b), k=tensor(64,6,t,b), v=k;
        auto mask=tensor(t,1,1,b), out=q;
        assert(managed_sdpa_decode::metadata(&q,&k,&v,&mask,&out,.125f,p));
        assert(p.tokens == uint32_t(t) && p.batches == uint32_t(b));
        assert(p.k_head_stride==64 && p.k_token_stride==384 && p.k_batch_stride==uint32_t(384*t));
        assert(p.out_head_stride==64 && p.out_batch_stride==384);
        ++passed;
    }
    auto q=tensor(64,6,1,2), k=tensor(64,6,1500,2), v=k;
    auto mask=tensor(1500,1,1,1), out=q;
    // Token padding and borrowed SET_ROWS-produced aliases remain valid.
    k.nb[2]+=32; k.nb[3]=k.nb[2]*1500; v=k;
    auto backing=k; k.op=GGML_OP_SET_ROWS; k.view_src=&backing;
    assert(managed_sdpa_decode::metadata(&q,&k,&v,&mask,&out,.125f,p));
    assert(p.mask_batch_stride==0 && p.k_token_stride==392); ++passed;
    auto reject=[&](const ggml_tensor & badq, const ggml_tensor & badk,
                    const ggml_tensor & badv, const ggml_tensor & badmask,
                    const ggml_tensor & badout, float scale=.125f) {
        const bool accepted=managed_sdpa_decode::metadata(&badq,&badk,&badv,&badmask,&badout,scale,p);
        if (accepted) std::cerr << "unexpected accepted rejection control index=" << passed << "\n";
        assert(!accepted); ++passed;
    };
    auto bad=q; bad.ne[0]=32; reject(bad,k,v,mask,out);
    bad=q; bad.ne[2]=2; reject(bad,k,v,mask,out);
    bad=q; bad.ne[3]=3; reject(bad,k,v,mask,out);
    bad=k; bad.ne[2]=1537; reject(q,bad,v,mask,out);
    bad=k; bad.ne[1]=5; reject(q,bad,v,mask,out);
    bad=v; bad.type=GGML_TYPE_F16; reject(q,k,bad,mask,out);
    bad=v; bad.nb[2]=64*4; reject(q,k,bad,mask,out);
    bad=v; bad.nb[3]=bad.nb[2]; reject(q,k,bad,mask,out);
    bad=mask; bad.nb[0]=8; reject(q,k,v,bad,out);
    bad=mask; bad.ne[0]=1499; reject(q,k,v,bad,out);
    bad=out; bad.view_src=&out; reject(q,k,v,mask,bad);
    bad=out; bad.nb[1]+=4; reject(q,k,v,mask,bad);
    bad=k; bad.nb[2]=UINT32_MAX; reject(q,bad,v,mask,out);
    bad=k; bad.nb[3]=UINT32_MAX-3; reject(q,bad,v,mask,out);
    reject(q,k,v,mask,out,__builtin_inff());
    std::cout << "{\"status\":\"PASS\",\"metadata_controls\":" << passed
              << ",\"scope\":\"host metadata only; no native context\"}\n";
}
