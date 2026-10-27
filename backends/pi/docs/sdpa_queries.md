# Bounded F32 multi-query attention experiment

This new operator leaves `sdpa_decode_f32` and its shader/metadata byte-identical.
It is an unqualified physical experiment, not an expansion of a model's logical
support or a replacement for the retained F32 attention graph.

## Frozen integration interface

`Graph.sdpa_queries_f32(q, k, v, mask, scale)` returns compact F32
`[64,H,Q,B]` (ggml fastest-first dimensions). Inputs are F32
`q[64,H,Q,B]`, `k/v[64,H,T,B]`, and `mask[T,Q or1,1,B or1]`.
The bounded physical domain is `1<=Q<=4`, `1<=T<=1536`, `1<=B<=2`, `H>=1`,
finite F32 scale, and checked non-overlapping head/query/token/batch strides.
Query and batch mask broadcasting are explicit zero strides in the ABI.

`Graph.supports_sdpa_queries_f32(heads,tokens,queries,batch=1)` is a
preparation-only query on the actual current Runtime DRM device, following the
existing Q1 capability query. Invalid metadata/operational errors propagate;
only the dedicated unavailable-capability result is false. Python selection
uses the new operation only for `Q>1`, and missing/false support retains the
original GPU graph before preparation. No post-submission fallback exists.

The native owner's common integration must use this exact header's metadata
routine in both preparation and record, recognize an actual distinct CUSTOM
identity, retain the four original producer pointers (including SET_ROWS and
GET_ROWS aliases), and reject unknown callback/task/userdata/extra-source state.
`PreparedGraph.info()` should expose
`managed_sdpa_queries_nodes_per_slot` using actual CUSTOM identity.

Registration name and new shader filename are `managed_sdpa_queries_f32` and
`vulkan-shaders/managed_sdpa_queries_f32.comp`. Binding order is q,k,v,mask,out.
The push constants are exactly `managed_sdpa_queries::Parameters`, 76 bytes,
with F32-element strides. Dispatch is `(H,B,Q)`, local size `(256,1,1)`.
The common helper must check subgroup16, WG256, arithmetic subgroup support,
11,264 bytes of shared memory, 76 bytes of push constants, and all three actual
dispatch-axis limits. Descriptor alignment/range must be checked against the
selected device and exact canonical views; output/input overlap is rejected.

No native private allocation, scratch BO, private KV, host work, or additional
stream/tail/lease owner is introduced. The output is an ordinary canonical arena
intermediate or explicit destination. Existing range barriers and retirement
cover it. Input view byte offsets are already incorporated in checked descriptor
subbuffers; push constants remain relative to those views.

## Algorithm and evidence scope

Each `(head,batch,query)` workgroup executes the accepted Q1 score, WG256
softmax, and coalesced original-V weighted sum. Only query indexing/strides and
dispatch metadata differ. K/V are reread independently for each query: this
first experiment deliberately makes no query-reuse or full-encoder claim.
Every physical token remains in the score/weighted-value domain, including
masked nonfinite values. All-minus-infinity masks retain the explicit equation's
NaN result. The old Q1 shader's exact arithmetic/reduction order is preserved
within each query; primitive/model numerical policy remains unchanged.

For Q4,H6,T1500, the old path materializes each K and transposed V as 2,304,000
bytes plus 144,000-byte score and probability tensors. The new operator removes
these intermediates and copies, but its actual completion time must be compared
with the original Q4 graph under the same current binary. Prior measured Q1
13.54ms/49.99ms versus 0.594ms/1.373ms is mechanism motivation only; Q4 is not
measured. Newly faster matrix paths make historical Q4 timing unsuitable as an
acceptance baseline.

Full encoder Q1500 has a different reuse tradeoff. Independently scanning K/V
for every query could lose the staged MM's query reuse. It also pre-scales Q
before the score product, unlike decoder attention. This experiment therefore
does not select the new operator for encoder attention.
