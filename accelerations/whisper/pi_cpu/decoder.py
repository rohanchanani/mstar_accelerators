"""Pi 5 CPU decoder: Q8 projections, Q4 KleidiAI readout and owner-held FP16 KV."""
from functools import partial

import torch
from mstar.engine.resources.sampler.config import GREEDY_TEMPERATURE
from mstar.model.whisper.submodules import WhisperDecoderSubmodule
from mstar_accelerators import Acceleration, AllRows, Eq, Implementation, Or, Range, When
from mstar.engine.resources.sampler.cpu import categorical_token
from mstar_accelerators.implementations.whisper.decoder_call import Call, Inputs, Plans, Resources
from mstar_accelerators.implementations.whisper.cpu_ggml_decoder import CPUTokenDecoderAvailable, prepare, resource_requests

common = When(
    Eq('node_name', 'decoder'),
    Eq('device_type', 'cpu'),
    Eq('dtype', torch.float32),
    Eq('tp_size', 1),
    Eq('sp_size', 1),
    Or(
        When(Eq('graph_walk', 'decode'), Eq('token_lengths', (1,))),
        When(Eq('graph_walk', 'prefill'), Eq('token_lengths', (4,))),
    ),
    Eq('batch_size', 1),
    Eq('config.d_model', 384),
    Eq('config.decoder_layers', 4),
    Eq('config.decoder_attention_heads', 6),
    Eq('config.decoder_ffn_dim', 1536),
    Eq('config.max_source_positions', 1500),
    Eq('config.max_target_positions', 448),
    Eq('config.vocab_size', 51865),
    Eq('config.activation_function', 'gelu'),
    Eq('config.scale_embedding', False),
)

# The runtime's CPU consumer implements the sampler policy, including the
# upstream generation controls. The native argmax remains the plain greedy path.
sampling_policy = AllRows(Resources.sampler, Or(
    Eq('temperature', 0.0),
    Range('temperature', GREEDY_TEMPERATURE, float(torch.finfo(torch.float32).max)),
))

acceleration = Acceleration(
    reference=WhisperDecoderSubmodule,
    when=When(common, sampling_policy, CPUTokenDecoderAvailable(readout_format="q4_0_kai")),
    implementation=Implementation(
        prepare=partial(prepare, sample_tokens=categorical_token, readout_format="q4_0_kai"),
        resources=resource_requests,
    ),
)
