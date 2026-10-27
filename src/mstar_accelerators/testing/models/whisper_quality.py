"""Whisper task continuations: one candidate node, otherwise reference M* engines.

The harness chooses cases and copies committed fixture state. Engine still owns
admission, allocation, plans, execution and commit, including reference prefill.
No optimizer supplies continuation code or acceptance policy.
"""
from dataclasses import replace
from types import SimpleNamespace
import torch

from mstar.engine.execution import ExecutionCapabilities, ExecutionOption, InvocationFacts, Support
from mstar.engine.resources.kv.cache import KVCache
from mstar.engine.resources.sampler.config import SamplingReqConfig
from mstar.model.binding import NodeAssembly
from mstar.model.whisper.config import KV_CACHE, CROSS_KV_CACHE, POS, SAMPLER
from mstar.model.whisper.results import DecoderResult
from mstar.model.whisper.submodules import WhisperEncoderSubmodule
from mstar_accelerators.registry import Registry
from mstar_accelerators.binding import static_facts, invocation_facts
from ..harness import Comparison, Unavailable
from ..quality import QualityOutput, fixture_path
from ..runtime import load_engine, join_placements, execute
from ..whisper import audio_inputs
from .whisper_reference import WhisperReference


def supported_walks(acceleration, binding, spec):
    """Validate the fixed quality protocol against when before any execution."""
    from ..sampling import request_config
    # Reuse the platform's canonical policy facts; no forced out-of-domain run.
    from ..sampling import policy_domains
    prompt = binding.model.config.decoder_prompt_ids(spec['language'], spec['task'])
    encoder = acceleration.reference is WhisperEncoderSubmodule
    walks = []
    policies = policy_domains(acceleration.when, SAMPLER)
    for walk in ('prefill',) if encoder else ('prefill', 'decode'):
        for policy in ({},) if encoder else policies:
            if not encoder and request_config({'resources': policy}, SAMPLER).temperature != 0:
                continue
            lengths = None if encoder else ((len(prompt),) if walk == 'prefill' else (1,))
            facts = {**static_facts(binding), **invocation_facts(InvocationFacts(
                walk, 1, lengths, policy, row_walks=(walk,)))}
            if acceleration.when.evaluate(facts) is Support.YES:
                walks.append(walk)
                break
    if not walks:
        raise Unavailable('quality_domain', 'fixed greedy quality protocol is outside this registration support')
    if spec['temperature'] != 0 or spec['task'] != 'transcribe' or spec['language'] != 'en':
        raise Unavailable('quality_protocol', 'central continuation currently supports English greedy transcription')
    return tuple(walks)


def _copy_committed(adapter, source, source_plans, target, target_plans):
    """Copy logical tokens via each owner's addressing, never assume page IDs match."""
    roots = []
    for key in (KV_CACHE, CROSS_KV_CACHE):
        src, dst = source[key].kv_cache, target[key].kv_cache
        src_cpu = KVCache(src.config, torch.device('cpu'), src.dtype,
                          storage=adapter._observe({'state': src.tensor})['state'])
        dst_cpu = KVCache(dst.config, torch.device('cpu'), dst.dtype,
                          storage=adapter._observe({'state': dst.tensor})['state'])
        if set(source_plans[key]) != set(target_plans[key]):
            raise RuntimeError('continuation logical KV labels differ')
        for label in source_plans[key]:
            a = source_plans[key][label].write_locations
            b = target_plans[key][label].write_locations
            if a.token_to_page.numel() != b.token_to_page.numel():
                raise RuntimeError('continuation logical KV write lengths differ')
            ai = (a.token_to_page.cpu(), a.token_to_cache.cpu())
            bi = (b.token_to_page.cpu(), b.token_to_cache.cpu())
            for layer in range(src.num_layers):
                values = src_cpu.read_tokens(layer, *ai)
                if not bool(values.isfinite().all()):
                    raise RuntimeError('nonfinite continuation state')
                dst_cpu.write_tokens(layer, values[:, 0], values[:, 1], *bi)
        roots.append(dst_cpu.tensor)
        dst.tensor.copy_(dst_cpu.tensor, non_blocking=True)
    adapter.session.retained.extend(roots)
    if adapter.stream is not None:
        adapter.stream.synchronize()


def _reference_prefill(adapter, reference):
    """Ordinary Engine callable for reference-created initial state of decode-only cases."""
    def bind(resources):
        def run(invocation):
            ids = invocation.inputs[0].input_ids.detach().cpu()
            states = invocation.inputs[0].tensor_inputs['encoder_states'].detach().cpu()
            batch, result = reference.invoke('prefill', 0, ids, states)
            _copy_committed(adapter, reference.resources, batch.step_context.plan_results,
                            resources, invocation.plan_results)
            return DecoderResult(result.tokens.to(adapter.device))
        return ExecutionCapabilities((ExecutionOption(
            name='quality-reference-prefill',
            supports=lambda facts: Support.YES if facts.graph_walk == 'prefill' else Support.NO,
            acquire=lambda facts: SimpleNamespace(close=lambda: None),
            materialize=lambda invocation, execution: invocation,
            primitive=lambda invocation, execution: run(invocation)),), {'prefill': 1})
    return NodeAssembly(bind=bind, resource_requests={}, execution_names=('quality-reference-prefill',))


@torch.no_grad()
def run_quality(adapter, acceleration, binding, spec, walks, *, encoder_acceleration=None):
    """One six-clip continuation per registration, independent of numerical witnesses."""
    encoder = acceleration is not None and acceleration.reference is WhisperEncoderSubmodule
    candidate_decoder = acceleration is not None and not encoder
    model = adapter.model
    prompt = torch.tensor(model.config.decoder_prompt_ids(spec['language'], spec['task']), dtype=torch.long)
    decoder_binding = replace(binding, node_name='decoder', submodule=model.create_submodule('decoder'),
                              device=torch.device('cpu'), dtype=torch.float32, prepared={})
    reference_decoder = WhisperReference(decoder_binding, adapter._reference(decoder_binding), adapter.session)
    encoder_binding = replace(decoder_binding, node_name='audio_encoder', submodule=model.create_submodule('audio_encoder'), prepared={})
    reference_encoder = WhisperReference(encoder_binding, adapter._reference(encoder_binding), adapter.session)
    engine = None
    if acceleration is not None:
        selected_binding = replace(binding, graph_walks=walks, prepared={})
        assembly = Registry({'candidate': acceleration}).assemble(selected_binding, require_coverage=False)
        if candidate_decoder and 'prefill' not in walks:
            assembly = join_placements(assembly, _reference_prefill(adapter, reference_decoder))
        engine = load_engine(selected_binding, assembly, adapter.session, adapter.parallel_groups)
    composed_encoder = None
    if encoder_acceleration is not None:
        selected_encoder = replace(binding, node_name='audio_encoder', submodule=model.create_submodule('audio_encoder'),
                                   graph_walks=('prefill',), prepared={})
        supported_walks(encoder_acceleration, selected_encoder, spec)
        enc_assembly = Registry({'candidate': encoder_acceleration}).assemble(selected_encoder, require_coverage=False)
        composed_encoder = load_engine(selected_encoder, enc_assembly, adapter.session, adapter.parallel_groups)
    rows = []
    for sample in spec['samples']:
        inputs, _ = audio_inputs(model, str(fixture_path(sample)), language=spec['language'], task=spec['task'])
        features = inputs['audio_features'][0]
        if encoder or composed_encoder is not None:
            encoder_engine = engine if encoder else composed_encoder
            encoder_engine.add_request(0)
            batch, observations = execute(encoder_engine, 'audio_encoder', 'prefill', 0, {'audio_features': [features]})
            if batch.admit_error is not None:
                raise RuntimeError(f'quality encoder admission: {batch.admit_error}')
            states = adapter._observe({'states': observations[0][0].encoder_states[0]})['states']
            encoder_engine.remove_request(0)
        else:
            _, result = reference_encoder.invoke('prefill', 0, features=features)
            states = result.encoder_states[0]
            reference_encoder.remove(0)
        if not bool(states.isfinite().all()):
            raise RuntimeError('nonfinite quality encoder output')
        ids, generated, stop = prompt, [], 'length'
        if candidate_decoder:
            cfg = SamplingReqConfig(temperature=0., repetition_penalty=1.)
            cfg.apply_conductor_config(seed=0, resource_key=SAMPLER)
            engine.add_request(0, {SAMPLER: cfg})
        for step in range(spec['max_new_tokens']):
            walk = 'prefill' if step == 0 else 'decode'
            candidate_step = candidate_decoder and (walk in walks or step == 0)
            if candidate_step:
                raw = {'text_inputs': [ids]}
                if step == 0:
                    raw['encoder_states'] = [states]
                adapter.graph_observation.values.clear()
                with adapter.graph_observation.capture_torch(model.config.vocab_size):
                    batch, observations = execute(engine, binding.node_name, walk, 0, raw)
                if batch.admit_error is not None:
                    raise RuntimeError(f'quality decoder admission: {batch.admit_error}')
                result = observations[0][0]
                token_tensor = adapter._observe({'tokens': result.tokens})['tokens']
                expected_position = len(prompt) + step
                if engine._resources[POS].position(0, 'main') != expected_position:
                    raise RuntimeError('quality position commit mismatch')
                if engine._resources[SAMPLER]._sampler._step_offset[0] != step + 1:
                    raise RuntimeError('quality sampler commit mismatch')
                if walk in walks:
                    if not adapter.graph_observation.values:
                        raise RuntimeError('missing quality sampler logits')
                    logits = adapter._observe({'logits': adapter.graph_observation.values[-1]})['logits']
                    mask = torch.zeros_like(logits, dtype=torch.bool)
                    mask[..., model.config.suppressed_tokens(walk)] = True
                    if not torch.equal(torch.isneginf(logits), mask) or not bool(logits[~mask].isfinite().all()):
                        raise RuntimeError('invalid quality logits/suppression')
                    if not torch.equal(logits.argmax(-1).reshape(-1), token_tensor.reshape(-1)):
                        raise RuntimeError('quality output is not own-logit greedy sample')
                if 'decode' not in walks and step == 0:
                    ref_batch, _ = reference_decoder.invoke('prefill', 0, prompt, states)
                    _copy_committed(adapter, engine._resources, batch.step_context.plan_results,
                                    reference_decoder.resources, ref_batch.step_context.plan_results)
                    for key, value in ((POS, len(prompt)), (SAMPLER, 1)):
                        actual = (reference_decoder.resources[key].position(0, 'main') if key == POS else
                                  reference_decoder.resources[key]._sampler._step_offset[0])
                        if actual != value:
                            raise RuntimeError('reference continuation progress mismatch')
            else:
                _, result = reference_decoder.invoke(walk, 0, ids, states if step == 0 else None)
                token_tensor = result.tokens.detach().cpu().clone()
            if token_tensor.numel() != 1 or token_tensor.dtype != torch.int64:
                raise RuntimeError('invalid quality token shape/dtype')
            token = int(token_tensor.item())
            generated.append(token)
            if token == model.config.eos_token_id:
                stop = 'eos'
                break
            ids = token_tensor.reshape(1)
        reference_decoder.remove(0)
        if candidate_decoder:
            engine.remove_request(0)
        rows.append({'id': sample['id'], 'text': model.tokenizer.decode(generated, skip_special_tokens=True),
                     'tokens': generated, 'stop': stop})
    return Comparison('quality.reference_continuation', QualityOutput(tuple(rows)), spec)
