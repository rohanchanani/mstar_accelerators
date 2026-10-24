# AMD foundation A: physical program provenance

This directory owns the physical Talker program selected by the A acceleration.
It does not construct, subclass, or install the model's retained default Talker
implementation. MStar supplies the logical node, logical checkpoint identities,
resources, and execution mechanisms. The acceleration chooses this program and
its execution/resource recipe.

## Source and transfer

The numerical donor is MStar's `users/atindra/amd-native` at
`678f59d6adb3d892263f2ba30fcf7b28145abf2e`, carried into the supplied A integration
on the Package 3 base `e64288bebc1d87fbdab4480b6480e9c356a92d7c`. The supplied A
integration includes the corrected initialization and logical-input preparation
needed by the fork. Its physical algorithms are the baseline for this transfer.

| Local file | Donor source | Adaptation |
| --- | --- | --- |
| `model.py` | `mstar/model/qwen3_omni/components/talker.py` | Copy of all Talker and depth-predictor numerical classes. Model-specific attention resolves locally. Predictor kernel imports are lazy so importing/preparing the program on CPU does not require Triton. Arithmetic is unchanged. |
| `attention.py` | `mstar/model/qwen3_omni/components/attention.py`, Talker branch of `Qwen3OmniAttention` | Retains the standard 1-D QK-normalized attention path. The unrelated Thinker MRoPE path is omitted. |
| `program.py` | `TalkerSubmodule` physical methods from `mstar/model/qwen3_omni/submodules.py` | Owns text/hidden projections, special projected constants, packing, sampling, predictor scratch, and capture templates. Removes request interpretation and graph-edge routing. Emits logical output fields as batch-row tensors. |
| `preparation.py` | Supplied A `mstar/model/qwen3_omni/reference.py` and checkpoint name remapping | Owns fused QKV/expert parameter layouts, weight transfer, predictor-head stacking, and explicit Triton MoE selection. Uses the logical checkpoint source and generic weight-loader primitives; it does not call the model's physical factory or name-remapping method. |

## Deliberately selected shared primitives

This program reuses MStar's generic RMSNorm, parallel linear/attention/MLP
modules, Triton MoE implementation, and generic checkpoint loader. Its predictor
selects `mstar.utils.attention.fused_qk_norm_rope` and `decode_attn_nhd`. These
are reusable numerical/kernel primitives, not an inherited Qwen implementation.
The acceleration's resource declaration selects the Talker attention, sequence
storage, position, and inline-sampling implementations; the runtime owns their
admission, planning, authoritative state, commit, and release.

`prepare_program` copies the logical model configuration before setting
`talker_text.moe_backend = "triton"`. This selects the same kernel as the donor's
current `auto` behavior while keeping that physical decision local to A. It
does not modify the logical model configuration or defer the choice to a
deployment setting.

## Baseline behavior retained

All three walks are preserved: variable-length Talker prefill, the six-token
assistant prefix, and single-token streaming decode. Text and hidden projections
remain in `preprocess`, outside captured forward execution. Pad/BOS/EOS values
are projected one row at a time at initialization, as in the donor.

The program uses the deployment-selected BF16 parameters for A, QK normalization
and RoPE kernels in the same places, and an FP32 codec-embedding accumulator.
Talker sampling suppresses the special-token region except EOS and applies
repetition penalty through its inline sampler. The depth predictor makes one
inline draw per residual group: **15 draws for the published 16-group model**.
Its dense scratch is overwritten in depth order; attention reads only the valid
prefix. The logical shell owns the once-only EOS-conditioning transition,
stopping, request cleanup, and mapping of typed results to graph edges.

Physical results use `codec_tokens`, `codec_embeds`, and `tokens` row buffers.
The runtime removes padding, clones graph-owned results before the next replay,
adapts the fields to the logical result type, and invokes the model's semantic
output mapping. Inline sampler effects are not fabricated again as external
sampling results.

## What CPU evidence establishes

The focused tests load deterministic small safetensors checkpoints into the
actual copied modules, check packed weights/BF16 boundaries, and run complete
FP32 and BF16 request lifecycles with real KV/position/sampler owners at batches
1, 2, and 8.
CPU attention, RoPE, and sampling functions are explicitly installed by the
tests; they are diagnostic substitutes, not a serving fallback or GPU emulator.
An independent functional PyTorch oracle reads logical checkpoint weights and
checks greedy output codes, codec accumulation, every layer's logical KV, and
position/sampling history across all three walks with reordered requests. It
imports none of the copied numerical model, packed weights, or selected kernels.
Source review separately verifies the copied numerical class bodies against the
supplied A donor. These checks do not establish ROCm kernel correctness, graph
capture/replay, actual checkpoint speech quality, or AMD performance. Those
remain requirements of the hardware qualification run.
