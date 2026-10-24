# Acceleration source corpus

`whisper/pi_v3d` contains four returned books: the encoder, greedy paged prefill,
greedy contiguous decode, and combined positive-temperature decoder.
`whisper/cpu_tp2` contains an encoder and two combined decoder alternatives with
different FFN collective realizations. `qwen3_omni/amd_a` contains the
external Talker book for the AMD A starting point. Each result retains its own
target evidence; adding a file or discovering it is not qualification.

Start a new authoring exercise by running the installed CLI into a fresh
directory **outside this corpus**. Preserve the unedited generated files and the
generated-to-filled diff. Keep the generated logical reference, inputs,
resources, plans and call types. Choose an execution template explicitly when
using a runner other than the default callable; the target device does not make
that choice. Unused generated files are authoring receipts, not registrations.

Each Python file exports `acceleration`. Discovery uses its relative path as its
registration identity. A deliberate file copy adds a variant without editing an
index; specialize its conditions and implementation as needed. Ordinary shared
Python code belongs outside this corpus because every `.py` here is a declaration.

The empty corpus is valid. Its qualification verdict, like the verdict for a
generated placeholder, belongs to the selected testing consumer.
