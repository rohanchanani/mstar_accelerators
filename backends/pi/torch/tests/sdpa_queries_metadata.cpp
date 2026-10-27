// Host metadata controls only; no native context, model oracle or GPU evidence.
#include "managed-sdpa-queries.h"
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
    managed_sdpa_queries::Parameters p{};
    for (const int n : {1,2,3,4}) for (const int t : {1,17,448,1500,1536})
    for (const int b : {1,2}) for (const bool query_broadcast : {false,true})
    for (const bool batch_broadcast : {false,true}) {
        auto q=tensor(64,6,n,b), k=tensor(64,6,t,b), v=k;
        auto mask=tensor(t,query_broadcast ? 1 : n,1,batch_broadcast ? 1 : b), out=q;
        assert(managed_sdpa_queries::metadata(&q,&k,&v,&mask,&out,.125f,p));
        assert(p.queries==uint32_t(n) && p.tokens==uint32_t(t) && p.batches==uint32_t(b));
        assert(p.q_query_stride==384 && p.q_batch_stride==uint32_t(384*n));
        assert(p.mask_query_stride==(query_broadcast || n==1 ? 0u : uint32_t(t)));
        assert(p.mask_batch_stride==(batch_broadcast || b==1 ? 0u : uint32_t(t*(query_broadcast ? 1 : n))));
        assert(p.out_head_stride==64 && p.out_query_stride==384 && p.out_batch_stride==uint32_t(384*n));
        ++passed;
    }
    auto q=tensor(64,6,4,2), k=tensor(64,6,1500,2), v=k;
    auto mask=tensor(1500,4,1,2), out=q;
    // Independent padded token/head/query rows and real producer aliases.
    q.nb[1]+=16; q.nb[2]=q.nb[1]*6+32; q.nb[3]=q.nb[2]*4+64;
    k.nb[2]+=32; k.nb[3]=k.nb[2]*1500;
    v.nb[2]+=64; v.nb[3]=v.nb[2]*1500;
    mask.nb[1]+=32; mask.nb[3]=mask.nb[1]*4+64;
    auto backing=k; k.op=GGML_OP_SET_ROWS; k.view_src=&backing;
    assert(managed_sdpa_queries::metadata(&q,&k,&v,&mask,&out,.125f,p));
    assert(p.k_token_stride==392 && p.v_token_stride==400 && p.mask_query_stride==1508); ++passed;
    auto reject=[&](const ggml_tensor & badq, const ggml_tensor & badk,
                    const ggml_tensor & badv, const ggml_tensor & badmask,
                    const ggml_tensor & badout, float scale=.125f) {
        const bool accepted=managed_sdpa_queries::metadata(&badq,&badk,&badv,&badmask,&badout,scale,p);
        if (accepted) std::cerr << "unexpected accepted rejection control index=" << passed << "\n";
        assert(!accepted); ++passed;
    };
    auto bad=q; bad.ne[0]=32; reject(bad,k,v,mask,out);
    bad=q; bad.ne[2]=5; reject(bad,k,v,mask,out);
    bad=q; bad.nb[2]=256; reject(bad,k,v,mask,out);
    bad=q; bad.nb[3]=q.nb[2]; reject(bad,k,v,mask,out);
    bad=q; bad.ne[3]=3; reject(bad,k,v,mask,out);
    bad=k; bad.ne[2]=1537; reject(q,bad,v,mask,out);
    bad=k; bad.ne[1]=5; reject(q,bad,v,mask,out);
    bad=v; bad.type=GGML_TYPE_F16; reject(q,k,bad,mask,out);
    bad=v; bad.nb[2]=64*4; reject(q,k,bad,mask,out);
    bad=v; bad.nb[3]=bad.nb[2]; reject(q,k,bad,mask,out);
    bad=mask; bad.nb[0]=8; reject(q,k,v,bad,out);
    bad=mask; bad.ne[1]=3; reject(q,k,v,bad,out);
    bad=mask; bad.nb[1]=1499*4; reject(q,k,v,bad,out);
    bad=mask; bad.nb[3]=mask.nb[1]; reject(q,k,v,bad,out);
    bad=mask; bad.ne[0]=1499; reject(q,k,v,bad,out);
    bad=out; bad.view_src=&out; reject(q,k,v,mask,bad);
    bad=out; bad.nb[2]+=4; reject(q,k,v,mask,bad);
    bad=out; bad.nb[3]+=4; reject(q,k,v,mask,bad);
    bad=out; bad.nb[1]+=4; reject(q,k,v,mask,bad);
    bad=k; bad.nb[2]=UINT32_MAX; reject(q,bad,v,mask,out);
    bad=k; bad.nb[3]=UINT32_MAX-3; reject(q,bad,v,mask,out);
    reject(q,k,v,mask,out,__builtin_inff());
    std::cout << "{\"status\":\"PASS\",\"metadata_controls\":" << passed
              << ",\"scope\":\"host metadata only; no native context\"}\n";
}
