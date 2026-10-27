"""Stock HF node arithmetic, invoked only by M*'s callable execution runner.

Floating checkpoint values are promoted without changing their values. HF's
explicit precision casts remain intact. Large promoted models use file-backed CPU weights; this affects reference
placement, never candidate execution or policy.
"""
import json
import tempfile
import math
from pathlib import Path
import torch
from torch import nn
from mstar.model.qwen3_omni.config import THINKER_KV, THINKER_SAMPLER
from mstar.model.qwen3_omni.results import ThinkerResult, AudioResult
from ..reference_execution import ReferenceExecution
from ..reference_resources import greedy_sampler
from .qwen_runtime_reference import reference_binding, read_history, write_history


class _WeightFiles:
    """File-backed CPU parameter storage; the OS owns paging, not M*."""
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory(prefix='mstar-reference-')
        self.count = 0

    def empty(self, shape, dtype):
        path = Path(self.directory.name) / str(self.count)
        self.count += 1
        return torch.from_file(str(path), shared=True, size=math.prod(shape), dtype=dtype).reshape(shape)

    def close(self):
        self.directory.cleanup()


def _load(module, source, prefix, device, dtype, *, storage=None):
    weights = source.load(n for n in source.by_name if n.startswith(prefix))
    for name, parameter in list(module.named_parameters()):
        logical = prefix + name
        probe = logical
        if name.endswith('.experts.gate_up_proj'):
            probe = logical.removesuffix('.gate_up_proj') + '.0.gate_proj.weight'
        elif name.endswith('.experts.down_proj'):
            probe = logical.removesuffix('.down_proj') + '.0.down_proj.weight'
        if probe not in weights:
            logical = logical.replace('.mlp.', '.block_sparse_moe.')
        target = device
        def empty():
            return (storage.empty(parameter.shape, dtype) if storage is not None else
                    torch.empty(parameter.shape, device=target, dtype=dtype))
        if name.endswith('.experts.gate_up_proj'):
            stem = logical.removesuffix('.gate_up_proj')
            value = empty()
            for i in range(value.shape[0]):
                value[i].copy_(torch.cat([weights.pop(f'{stem}.{i}.{part}_proj.weight')
                                          for part in ('gate', 'up')]))
        elif name.endswith('.experts.down_proj'):
            stem = logical.removesuffix('.down_proj')
            value = empty()
            for i in range(value.shape[0]):
                value[i].copy_(weights.pop(f'{stem}.{i}.down_proj.weight'))
        else:
            value = empty().copy_(weights.pop(logical))
        if value.shape != parameter.shape:
            raise ValueError(f'checkpoint shape mismatch: {logical}')
        parent, key = name.rsplit('.', 1)
        module.get_submodule(parent).register_parameter(key, nn.Parameter(value, requires_grad=False))
    if weights:
        raise ValueError(f'unconsumed reference weights: {list(weights)[:5]}')


class NodeReference:
    def __init__(self, model, node, session, device, dtype, walks):
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeThinkerTextModel, Qwen3OmniMoeAudioEncoder)
        self.model, self.node, self.dtype = model, node, dtype
        self.device = torch.device(device)
        config = Qwen3OmniMoeConfig(**json.loads((Path(model.local_dir)/'config.json').read_text())).thinker_config
        source = model.logical_weights()
        if node == 'Thinker':
            cfg = config.text_config
            cfg._attn_implementation = cfg._experts_implementation = 'eager'
            with torch.device('meta'):
                self.module = Qwen3OmniMoeThinkerTextModel(cfg)
            self.storage = _WeightFiles() if self.device.type == 'cpu' else None
            _load(self.module, source, 'thinker.model.', self.device, dtype, storage=self.storage)
            self.module.rotary_emb = type(self.module.rotary_emb)(cfg, device=self.device)
            self.head = source.load(('thinker.lm_head.weight',))['thinker.lm_head.weight'].to(self.device, dtype)
            self.module.layers[model.config.accept_hidden_layer].register_forward_hook(self._condition)
            self.placement = {'device': str(self.device), 'file_backed_weights': self.storage is not None}
        else:
            self.storage = None
            cfg = config.audio_config
            cfg._attn_implementation = 'eager'
            # Audio's constructed positional buffers are part of its BF16 state.
            self.module = Qwen3OmniMoeAudioEncoder(cfg).to(torch.bfloat16).to(dtype)
            _load(self.module, source, 'thinker.audio_tower.', self.device, dtype)
            self.module.to(self.device)
            self.placement = {'device': str(self.device)}
        self.module.eval()
        self.binding = reference_binding(model, node, device, dtype, walks)
        self.execution = ReferenceExecution(self.binding, self, session,
                                            providers={THINKER_SAMPLER: greedy_sampler})

    def close(self):
        self.module = self.head = None
        if self.storage is not None:
            self.storage.close()

    def _condition(self, module, args, output):
        self.conditioning = (output[0] if isinstance(output, tuple) else output).clone()

    def bind_resources(self, resources): self.resources = resources

    def run(self, invocation):
        if self.node == 'audio_encoder':
            values = []
            for row in invocation.inputs:
                output = self.module(row.tensor_inputs['audio_features'].to(self.device, self.dtype),
                                     feature_lens=row.tensor_inputs['audio_seqlens'].to(self.device))
                values.append(output.last_hidden_state)
            return AudioResult(tuple(values))
        from transformers.cache_utils import DynamicCache
        cfg = self.module.config
        logits, conditioning, masks = [], [], []
        for index, row in enumerate(invocation.inputs):
            walk = invocation.row_walks[index] if invocation.row_walks else invocation.graph_walk
            if row.input_ids is not None:
                x = self.module.embed_tokens(row.input_ids)
            else:
                kind = 'audio' if walk == 'prefill_audio' else 'vision'
                ids = torch.tensor([getattr(self.model.config.thinker, kind+'_start_token_id'),
                                    getattr(self.model.config.thinker, kind+'_end_token_id')], device=self.device)
                sentinel = self.module.embed_tokens(ids)
                x = torch.cat((sentinel[:1], row.input_embeds.to(self.dtype), sentinel[1:]))
            cache = DynamicCache(config=cfg)
            for layer, pair in enumerate(read_history(self.resources[THINKER_KV], invocation, THINKER_KV, index)):
                if pair is not None:
                    cache.update(*(v.transpose(0,1)[None] for v in pair), layer)
            kwargs = {}
            if walk == 'prefill_vision':
                mask = row.tensor_inputs['masks_for_talker'][0]
                kwargs = dict(visual_pos_masks=mask[None],
                              deepstack_visual_embeds=[v[mask].to(self.dtype) for v in row.tensor_inputs['deepstack']])
            result = self.module(inputs_embeds=x[None], position_ids=row.custom_pos_ids[:,None,:],
                                 past_key_values=cache, use_cache=True, **kwargs)
            history = [(layer.keys[0].transpose(0,1), layer.values[0].transpose(0,1)) for layer in cache.layers]
            write_history(self.resources[THINKER_KV], invocation, THINKER_KV, index, history)
            conditioning.append(torch.cat((x, self.conditioning[0]), dim=-1))
            masks.append(row.tensor_inputs['masks_for_talker'])
            logits.append(torch.nn.functional.linear(result.last_hidden_state[0,-1], self.head))
        tokens = self.resources[THINKER_SAMPLER].sample(invocation.request_ids, logits=torch.stack(logits))
        return ThinkerResult(tuple(token.reshape(1) for token in tokens), tuple(conditioning), tuple(masks))
