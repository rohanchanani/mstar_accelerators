"""Request-local audio attention lengths for the qualified Transformers 5.10.2.

The encoder body mirrors that pinned HF forward to thread one exact integer
through its existing layer kwargs. Keep this copy in sync before widening the
version guard. Unsupported modes use the original HF implementation, including
its output-capture decorators. No global or pointer-keyed metadata cache exists.
"""

import torch
import transformers
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf


class AudioAttentionWithLengths(hf.Qwen3OmniMoeAudioAttention):
    def forward(self, hidden_states, cu_seqlens, *, audio_max_length=None, **kwargs):
        if audio_max_length is None or not hf.is_flash_attention_requested(self.config):
            return super().forward(hidden_states, cu_seqlens, **kwargs)
        seq_length = hidden_states.shape[0]
        query = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        key = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        attention = hf.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, hf.eager_attention_forward
        )
        # Preserve the original tensor-to-int boundary before varlen attention.
        # Hoisting metadata must not fuse across the existing BF16 materializations.
        torch._dynamo.graph_break()
        output, _ = attention(
            self,
            query,
            key,
            value,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=audio_max_length,
            max_length_k=audio_max_length,
            is_causal=False,
            **kwargs,
        )
        return self.out_proj(output.reshape(seq_length, -1).contiguous())


class AudioEncoderWithLengths(hf.Qwen3OmniMoeAudioEncoder):
    metadata_enabled = True

    def __init__(self, config):
        super().__init__(config)
        # Construct before checkpoint loading, keeping identical parameter names.
        if transformers.__version__ == "5.10.2":
            for layer in self.layers:
                layer.self_attn = AudioAttentionWithLengths(config)

    def _metadata_supported(self, input_features, kwargs):
        return (
            self.metadata_enabled
            and transformers.__version__ == "5.10.2"
            and not self.training
            and not torch.is_grad_enabled()
            and bool(torch.version.hip)
            and input_features is not None
            and input_features.is_cuda
            and hf.is_flash_attention_requested(self.config)
            and kwargs.get("return_dict", True) is not False
            and not getattr(self.config, "output_hidden_states", False)
            and not getattr(self.config, "output_attentions", False)
            and getattr(self.config, "return_dict", True)
            and set(kwargs).issubset({"return_dict", "padded_feature", "chunk_lengths", "valid_indices", "cu_seqlens"})
            and all(
                type(layer) is hf.Qwen3OmniMoeAudioEncoderLayer and type(layer.self_attn) is AudioAttentionWithLengths
                for layer in self.layers
            )
        )

    def forward(self, input_features=None, feature_lens=None, **kwargs):
        if not self._metadata_supported(input_features, kwargs):
            return super().forward(input_features, feature_lens, **kwargs)
        kwargs.pop("return_dict", None)
        padded_feature, chunk_lengths = hf.chunk_and_pad_features(
            input_features, feature_lens, self.n_window, kwargs=kwargs
        )
        valid_indices = hf.get_valid_indices(chunk_lengths, kwargs=kwargs)
        cu_seqlens = hf.get_audio_cu_seqlens(
            chunk_lengths, feature_lens, self.n_window_infer, self.n_window, kwargs=kwargs
        )
        # The same immutable lengths govern every layer of this invocation.
        max_length = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        padded_feature = padded_feature.unsqueeze(1)
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = torch.nn.functional.gelu(self.conv2d1(chunk))
            padded_embed = torch.nn.functional.gelu(self.conv2d2(padded_embed))
            padded_embed = torch.nn.functional.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
        position = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + position
        hidden_states = torch.index_select(padded_embed.reshape(-1, padded_embed.shape[-1]), 0, valid_indices)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens, audio_max_length=max_length)[0]
        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hf.BaseModelOutputWithPooling(last_hidden_state=hidden_states)
