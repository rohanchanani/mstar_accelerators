The Q8 encoder prefix-copy construction follows
`ggml-org/whisper.cpp` (MIT), pinned at
`d1be6fde11ac6e0407606b4e42fe72d34add8037` by the retained campaign source lock.
See `whisper_build_graph_encoder`: F32 projected K/V prefixes are copied into
prezeroed F16 roots before unmasked attention over the padded 256-position extent.
The copied prefix has exactly `width * positions` elements; no output is cropped.

This component retains both scratch roots in its existing graph arena. Input flags
allocate roots before intermediates; output flags prohibit gallocr reuse after the
last attention. They are not decoder state, do not persist meaningful request data
between executions, and are never exported. Their zero tails are initialized once;
every prefix is overwritten before each use under the existing synchronous slot
BusyGuard. The same owner destroys them with the graph arena. No new ABI, allocator,
request resource, state write or cross-KV preparation is introduced.

Public Whisper omits key bias. Here only an explicitly supplied all-positive-zero
bias is omitted; nonzero values, negative zero and nonfinite values retain ADD.
The F32 implementation retains its original graph equation.

`encoder_public_scratch.cpp` compiles the actual Encoder::attention implementation
against pinned GGML. It checks two layers sharing the roots across four executions,
finite bitwise equality to PAD-F32-to-F16 attention, nonfinite output classification,
zero-tail preservation under gallocr allocation, and bias-elision bit guards. This
is a host mechanism control, not target model qualification or performance evidence.
