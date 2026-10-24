# Authored CPU TP2 implementations

This directory contains the encoder and two combined decoder variants authored
from the installed CLI. The exact Package 3 generated starting files remain in
`authoring/package3_cli_generated` at the repository root. New experiments should
generate into a fresh directory outside corpus discovery and preserve their own
generated-to-filled diff.

`audio_encoder.py` prepares FP64 weights and arithmetic with FP32 logical
outputs. `decoder_all_reduce.py` and `decoder_gather_sum.py` partition the FFN
intermediate channels, then choose all-reduce or all-gather followed by local
summation. Attention and FP32 KV remain replicated, and both decoder books retain
the generated reference and call records. Their explicit costs resolve overlap.
These are ordinary callable executions; collective choice does not introduce a
new runner. See `docs/TORCH_IMPLEMENTATIONS.md` at the repository root for the
helper, resource, and ownership boundaries.
