# Optional native categorical graph operation

Package 2 adds `Graph.categorical(logits, seeds, offsets, temperatures)` to the
optional ggml implementation surface. It does not add a model-facing sampler,
request owner, or qualification entry point. An acceleration may use another
correct categorical implementation. M* owns the sampling configuration, seed and
logical offset, and advances that offset only when its retained-row commit
succeeds. The native operation only reads its invocation inputs.

## Interface and declared domain

| Binding | ggml dimensions | Storage |
| --- | --- | --- |
| `logits` | `[vocabulary, rows]` | contiguous F32 |
| `seeds` | `[rows]` | contiguous I64, interpreted as full 64-bit words |
| `offsets` | `[rows]` | contiguous I64, interpreted as full 64-bit words |
| `temperatures` | `[rows]` | contiguous F32 |
| return | `[rows]` | contiguous I64 token IDs |

The Torch row-major logits shape is `[rows, vocabulary]`. The primitive accepts
positive finite F32 temperatures. The serving declaration uses M*'s existing
canonical `GREEDY_TEMPERATURE` boundary, inclusive `1e-5`, through the F32 maximum
`(2-2**-23)*2**127`. The owner already normalizes raw temperatures below `1e-5`
to greedy zero; this backend does not invent another sampling threshold. A
configuration that would underflow or overflow F32 must be rejected before
native execution. Ordinary rounding, such as Python `0.8` to F32, is supported
and is part of this implementation's numerical policy. The primitive applies
temperature only; it does not apply top-k, top-p,
min-p, repetition penalties or model-specific suppression. The Package 2
acceleration admits canonical `top_k=0`, `top_p=1`, `min_p=0`, and
`repetition_penalty=1`; model suppression remains in the authored math graph.

Ordinary inputs contain finite logits with optional negative-infinity masks and
at least one unmasked entry per row. Metadata construction and preparation check
types, contiguity, vector lengths, index extents and descriptor alignment before
submission. Vocabulary is at most `INT32_MAX`, and total logits elements are at
most `UINT32_MAX`; actual prepared storage must also satisfy the device's normal
descriptor/allocation constraints. These metadata limits do not assert that an
arbitrary graph or allocation fits the Pi.

Exceptional row behavior is explicit and covered by the backend gate:

- Positive infinities restrict selection to those positive-infinity entries,
  with equal probability among them.
- A NaN logit, an all-negative-infinity row, or a nonpositive/nonfinite
  temperature yields I64 `-1` for that row.
- A finite equal-score tie chooses the lowest original vocabulary index.

The `-1` result is an invalid-sample sentinel outside the declared serving
domain. It does not authorize a successful token, fallback, state commit, or
transparent repair. The shader does not read logits back to the CPU to discover
these conditions. No arbitrary-input qualification follows from documenting
this substrate behavior.

## Reproducible counter mapping

The primitive's algorithm identifier is **`philox4x32-10-gumbel23-v1`**. This is
an implementation policy, not a mandatory RNG algorithm for every equivalent
categorical acceleration.

For vocabulary index `j`, split each input seed and offset into low/high 32-bit
words, preserving all bits. Invoke Philox4x32-10 with:

```
key     = (seed_low, seed_high)
counter = (offset_low, offset_high, j, 0)
```

The row's batch index is absent. Moving a request between rows cannot change its
draw. One logical offset selects one complete categorical draw, regardless of
vocabulary width or the number of Philox invocations used internally. Neither
this operation nor its prepared graph modifies seed/offset storage.

Each of ten Philox rounds uses unsigned 32-bit arithmetic:

```
(hi0, lo0) = full_product(0xD2511F53, counter[0])
(hi1, lo1) = full_product(0xCD9E8D57, counter[2])
counter   = (hi1 ^ counter[1] ^ key[0], lo1,
             hi0 ^ counter[3] ^ key[1], lo0)
key       = key + (0x9E3779B9, 0xBB67AE85)  modulo 2**32
```

The first output word supplies a strictly open uniform:

```
u = (float(word >> 9) + 0.5) * 2**-23
```

The 23-bit midpoint representation is exact in F32 and has endpoints `2**-24`
and `1-2**-24`. Using a 24-bit midpoint would allow the upper endpoint to round
to one, which is unsuitable for the nested logarithm. The finite grid and F32
arithmetic are numerical approximations; this is not a claim about arbitrary
tiny probabilities or cross-provider bit identity.

For finite logits, compute a row maximum and select the lowest-index maximizer
of `(logit[j] - row_max) / temperature - log(-log(u[j]))`. Equal row maxima use
centered value zero directly. Negative-infinity entries are excluded. The
Gumbel-max construction gives the temperature-scaled categorical law before
finite precision/finite uniform quantization. A fixed native implementation is
repeatable for the same inputs; transcendental differences can affect decisions
near a score boundary across implementations. Central qualification checks
sampling semantics and request effects independently of this exact RNG mapping.

The constants, round layout and ten-round definition follow the authors'
[Random123 source](https://www.random123.com/releases/docs/philox_8h_source.html).
The independent scalar test oracle checks these original known-answer vectors
from [Random123 tests/kat_vectors](https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors):

| Counter | Key | Four output words |
| --- | --- | --- |
| `00000000 00000000 00000000 00000000` | `00000000 00000000` | `6627e8d5 e169c58d bc57ac4c 9b00dbd8` |
| `ffffffff ffffffff ffffffff ffffffff` | `ffffffff ffffffff` | `408f276d 41c83b0e a20bc7c6 6d5451fd` |
| `243f6a88 85a308d3 13198a2e 03707344` | `a4093822 299f31d0` | `d16cfe09 94fdcceb 5001e420 24126ea1` |

The fetched primary vector file has SHA256
`aab5ebabf40003f63d6d87b24cbd2c8a02652e00cf8bad64226fd50586929183`.
The GLSL and scalar test implementation were written for this operation; no
Random123 runtime library or model-level RNG state is introduced.

## Execution and ownership

One 256-invocation workgroup handles each row. Lanes iterate over vocabulary
entries, reduce the row maximum, compute stateless Philox/Gumbel scores, and
reduce `(score, original_index)`. Workgroup shared memory holds two 256-element
arrays. `umulExtended` computes the full 32-bit product without `shaderInt64`.
I64 inputs and outputs use two-word storage, as the retained graph copy kernels
already do. The operation needs no subgroup feature or additional scratch BO.

The result intermediate uses the existing canonical graph arena; `Graph.output`
writes the ordinary caller-bound I64 token storage. The existing managed graph
descriptor preparation, input/output alias checks, synchronization, current
Torch stream, completion tail, allocation uses and retirement path are reused.
The implementation adds no queue, host continuation, host logit materialization,
global wait, CPU compute fallback, or independent cache/sampler owner.

## Build and evidence

Rebuild the userspace native/ggml libraries, the actual embedded shader bundle,
and the Torch extension (`mstar-pi` version `0.4.0`) from this source. Reusing the
Package 1 extension cannot provide this operation. The accepted kernel, Mesa/ICD,
DRM contract and admission protocol remain the target environment. A userspace
addition does not authorize kernel restaging or dependency substitution.

Host commands after that build:

```sh
python torch/tests/categorical_reference.py
python torch/tests/graph_host_preflight.py
python torch/tests/check_graph_spirv.py build/third_party/ggml/src/ggml-vulkan/vulkan-shaders.spv
```

The existing Stage 1, Stage 2 and four-case graph gates stay required and keep
their existing cases. The focused additional primitive gate shares the graph
gate's actual parent/spawn initialization, failure drain, storage-release health
checks, shutdown and parent join. These primitive processes do not register
`WorkerOwner` records; their gate uses the ordinary shared launcher shown here.
Managed worker acknowledgements remain part of the corpus/service worker path:

```sh
python tools/run_hardware.py --deadline 600 -- \
  python torch/tests/categorical_consumer.py \
    --shader build/shaders/transfer.spv --validation
```

It has three cases in each process: exact primitive RNG mapping with fixed
decision margins/repeat/offset/row-order/retained-result checks; categorical
distribution at two temperatures with explicit fixed binomial bounds and masks;
and exceptional values plus a fixed cross-lane score tie. Seed/offset/temperature
input immutability, guarded bound outputs, ordinary Torch clone/pinned-copy/event
consumption, slot retirement and final release are checked on the real path.
The source's three primary RNG vectors, shader ABI inspection, host metadata
execution, target backend conformance, whole-corpus registered acceleration
qualification and real service remain separate evidence classes. None is a
performance, overlap, arbitrary-input, other-device, or exceptional-reset claim.
