#pragma once
#include <stdint.h>
// Private physical marker in reserved FLASH_ATTN_EXT op_params word 4. Pinned
// ggml_new_tensor_impl initializes every op_params byte to zero. Only this
// explicitly built encoder variant writes it; decoder graphs stay unmarked.
#define MSTAR_ENCODER_QK_SDOT_TAG INT32_C(0x514b5344)
