"""Q1/D64/F16 physical chunk replacement; original planner/reducer unchanged."""

def ops_overlay(text, *, diagnostic=False):
    anchor = "static void ggml_compute_forward_flash_attn_ext_f16_one_chunk("
    if text.count(anchor) != 1 or '#include "flash_cross_q1.h"' in text:
        raise ValueError("cross-Q1 requires the pinned one-chunk entry exactly once")
    text = text.replace(anchor, '#include "flash_cross_q1.h"\n\n' + anchor)
    body = text.index("    const bool write_partials", text.index(anchor))
    call = "mstar_cross_q1_chunk(params,dst,ir0,ir1,ic_start,ic_end,partials,partial_stride)"
    if diagnostic:
        hook = ("    if (dst->op_params[15] == 1) {\n"
                "        if (" + call + ") { mstar_cross_fast.fetch_add(1, std::memory_order_relaxed); return; }\n"
                "        mstar_cross_fallback.fetch_add(1, std::memory_order_relaxed);\n    }\n")
        text = "#define MSTAR_CROSS_Q1_DIAGNOSTIC 1\n" + text
        body += len("#define MSTAR_CROSS_Q1_DIAGNOSTIC 1\n")
    else:
        hook = "    if (" + call + ") return;\n"
    return text[:body] + hook + text[body:]
