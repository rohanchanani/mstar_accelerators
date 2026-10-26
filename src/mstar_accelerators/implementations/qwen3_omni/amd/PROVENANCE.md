# AMD Talker source attribution

The physical Talker and predictor implementation derives from Atindra's
`users/atindra/amd-native` in MStar, commit
`678f59d6adb3d892263f2ba30fcf7b28145abf2e`.

The predictor embedding accumulation and QK normalization/RoPE/KV-write
kernels in `cp_kernels.py` are from Joseph Tan (`dghosef/mstar`), commit
`e2d13b5df022c1a4ce9a3df126d8ee8cfa108c76`, original blob
`0473119c42978b034f248715003d20ca4ee5f422`. Batched decode text projection
is adapted from his commit `3f10d2464ec881cf9b65bf8f18104d9b97f935f8`.

The acceleration owns physical numerical execution and weight preparation.
MStar owns logical request semantics, scheduling, resources and state lifetime.
The `mstar_amd_a::router_mm` dispatcher name is retained for compatibility;
it refers to the sole router implementation here, not a separate variant.
