"""Accelerator-owned audio encoder with request-local attention metadata."""
import inspect
import torch
from torch import nn
from transformers import AutoConfig
from mstar.model.components.hf_attention import build_hf_encoder
from mstar.model.qwen3_omni.components.attention import patch_hf_fa2_int_maxlen
from .audio_metadata import AudioEncoderWithLengths

class AudioProgram(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        params = inspect.signature(encoder.forward).parameters
        self.return_dict = "return_dict" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())

    def preprocess(self, graph_walk, engine_inputs, inputs):
        if len(inputs) != 1:
            raise ValueError("Audio encoder processes one logical request per invocation")
        return inputs[0].tensor_inputs

    def forward(self, graph_walk, engine_inputs, audio_features, audio_seqlens=None, **kwargs):
        extra = {"return_dict": True} if self.return_dict else {}
        out = self.encoder(audio_features, feature_lens=audio_seqlens, **extra)
        embeddings = getattr(out, "last_hidden_state", out)
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)
        return {"audio_embeds": [embeddings]}


def prepare_audio(binding, resources):
    del resources
    # HF configuration describes the logical architecture, not an implementation factory.
    config = AutoConfig.from_pretrained(binding.model.local_dir, trust_remote_code=False).thinker_config.audio_config
    patch_hf_fa2_int_maxlen()
    encoder = build_hf_encoder(AudioEncoderWithLengths, config)
    source = binding.model.logical_weights()
    values = source.load(name for name in source.by_name if name.startswith(("thinker.audio_tower.",)))
    encoder.load_state_dict({name.removeprefix("thinker.audio_tower."): value for name, value in values.items()})
    # The measured encoder retained FP32 weights with BF16 execution autocast.
    encoder.to(binding.device)
    return AudioProgram(encoder.eval()).eval()
