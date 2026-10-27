"""Talker-isolated, own-history speech continuations through ordinary M*.

Pure-text Thinker edges contain checkpoint token embeddings, not transformer
hidden states. Materialize those valid logical inputs once, then give both
Talker implementations the same conditioning. Each produces its own codes,
feedback, KV history and EOS. The shared downstream Code2Wav engine consumes
both streams using the model's actual chunk policy. No full service is started.
"""
from pathlib import Path
import tempfile
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import ExecutingBatch
from mstar.engine.execution import InvocationFacts, Support
from mstar.engine.resources.step import StepContext
from mstar.model.submodule_base import InputMetadata
from mstar.streaming.stream_buffer import StreamBuffer
from mstar.worker.engine_manager import EngineManager
from mstar_accelerators.binding import invocation_facts
from ..harness import Comparison, Unavailable
from ..speech import SpeechJudge, judge_path

# Fixed platform corpus, unrelated to acceleration names or implementations.
# The presentation case preserves the previously exposed opening regression.
CASES = (
    ('presentation', 'Give me one practical tip for preparing a clear presentation.',
     'Use bullet points sparingly and keep each slide focused on one key idea. '
     'Instead of overcrowding slides with text, limit each slide to one main concept '
     'and support it with concise bullet points or a single impactful image. '
     'This helps your audience follow along easily and keeps the presentation clear and engaging.'),
    ('short', 'Give me one practical tip for taking notes.',
     'Write down the main idea first, then add a few short details. '
     'Leave some space so you can add questions later.'),
)
MAX_FRAMES = 512
BATCHES = (1, 8)


def fixtures(model, dtype):
    judge = judge_path()  # Preflight before candidate submission.
    tokenizer = model.tokenizer
    source = model.logical_weights()
    name = 'thinker.model.embed_tokens.weight'
    table = source.load((name,))[name]
    shell = model.create_submodule('Thinker')
    values = []
    for key, prompt, text in CASES:
        ids = model.process_prompt(prompt=prompt, input_modalities=['text'],
                                   output_modalities=['audio'])['text_inputs'][0]
        # Same model-owned text inclusion semantics as the producer's
        # postprocess. This constructs a fixture; it runs no replacement model.
        included = shell._get_talker_text_mask(ids)
        prefill = table[ids[included]].to(dtype).clone()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        values.append({'id': key, 'text': text, 'prompt': prompt,
                       'prefill': prefill, 'text_embeddings': table[tokens].to(dtype).clone(),
                       'text_token_ids': tokens, 'prefill_token_ids': ids[included].tolist()})
    return values, judge


def _batch(engine, node, walk, raw, *, metadata=None, final=()):
    rids = list(raw)
    infos = {rid: CurrentForwardPassInfo(f'speech-{rid}', walk, 0, 0, MAX_FRAMES,
                 rid_handle=rid, step_metadata={'voice': 'Ethan', 'talker_max_tokens': MAX_FRAMES}) for rid in rids}
    batch = ExecutingBatch(node, infos, StepContext(rids, walk, 0, False),
                          per_request_input_tensors=raw,
                          per_request_input_metadata=metadata or {}, final_stream_rids=set(final))
    engine.prepare_inputs(batch)
    if batch.failed_requests or batch.skipped_rids:
        raise RuntimeError(f'speech inputs rejected: {batch.failed_requests}; {batch.skipped_rids}')
    return batch


def _candidate(adapter, name, acceleration, manager, cases, static, record):
    from .qwen3_omni import _policy
    engine = manager.engine
    eos = adapter.model.config.talker.codec_eos_token_id
    rid_cases = {100 + i: case for i, case in enumerate(cases)}
    previous, frames = {}, {rid: [] for rid in rid_cases}
    active = list(rid_cases)
    for rid in active:
        manager.add_request(rid, _policy(rid))
    def run(walk, raw):
        batch = _batch(engine, 'Talker', walk, raw)
        facts = static | invocation_facts(InvocationFacts(walk, len(raw),
            tuple(row.input_seq_len for row in batch.inputs),
            engine.resource_facts('Talker', list(raw)), row_walks=(walk,) * len(raw)))
        if acceleration.when.evaluate(facts) is not Support.YES:
            raise ValueError('speech continuation outside declared support')
        engine.reserve_replay_slot(batch)
        selected = batch.selected_execution
        if selected is None or selected.option.name != name:
            raise RuntimeError('speech continuation selected a different acceleration')
        result = engine.exec(batch)
        if batch.admit_error is not None or batch.failed_requests:
            raise RuntimeError(f'speech execution failed: {batch.admit_error}; {batch.failed_requests}')
        captured = batch.step_context.slot_lease is not None
        key = f'{walk}/b{len(raw)}/{"captured" if captured else "eager"}'
        record[key] = record.get(key, 0) + 1
        if walk != 'talker_prefill':
            for rid in raw:
                value = result.per_rid_outputs[rid]
                frames[rid].append(value['codec_tokens'][0].detach().cpu().clone().reshape(-1))
                previous[rid] = value['talker_input_embeds'][0]
    run('talker_prefill', {rid: {'thinker_states': [case['prefill'].to(adapter.device)],
        'thinker_mask': [torch.zeros(len(case['prefill']), dtype=torch.bool, device=adapter.device)]}
        for rid, case in rid_cases.items()})
    run('talker_last_prefill', {rid: {'thinker_states': [case['text_embeddings'][:1].to(adapter.device)]}
                               for rid, case in rid_cases.items()})
    stops = {}
    for index in range(1, MAX_FRAMES):
        for rid in tuple(active):
            if int(frames[rid][-1][0]) == eos:
                stops[rid] = 'eos'
                manager.remove_request(rid)
                active.remove(rid)
        if not active:
            break
        raw = {}
        for rid in active:
            raw[rid] = {'talker_input_embeds': [previous[rid]]}
            text = rid_cases[rid]['text_embeddings']
            if index < len(text):
                raw[rid]['thinker_states'] = [text[index:index+1].to(adapter.device)]
        run('talker_decode', raw)
    for rid in active:
        stops[rid] = 'eos' if int(frames[rid][-1][0]) == eos else 'limit'
        manager.remove_request(rid)
    return [(rid_cases[rid], torch.stack(frames[rid]), stops[rid]) for rid in rid_cases]


def _reference(adapter, case):
    oracle = adapter.oracle
    rid = 500
    oracle.add(rid)
    device = adapter.device
    oracle.step('talker_prefill', rid, {'thinker_states': [case['prefill'].to(device)],
        'thinker_mask': [torch.zeros(len(case['prefill']), dtype=torch.bool, device=device)]})
    frames = []
    previous = None
    for index in range(MAX_FRAMES):
        if index == 0:
            walk, raw = 'talker_last_prefill', {'thinker_states': [case['text_embeddings'][:1].to(device)]}
        else:
            walk, raw = 'talker_decode', {'talker_input_embeds': [previous]}
            if index < len(case['text_embeddings']):
                raw['thinker_states'] = [case['text_embeddings'][index:index+1].to(device)]
        value = oracle.step(walk, rid, raw)  # no teacher codes: own sampled history
        frames.append(value.result.codec_tokens.detach().cpu().reshape(-1))
        previous = value.result.codec_embeds
        if int(frames[-1][0]) == adapter.model.config.talker.codec_eos_token_id:
            break
    oracle.remove(rid)
    return torch.stack(frames), ('eos' if int(frames[-1][0]) == adapter.model.config.talker.codec_eos_token_id else 'limit')


def _decode(manager, model, device, frames, rid):
    connection = next(c for c in model.get_partition_topology().connections
                      if c.from_partition == 'Talker' and c.to_partition == 'Code2Wav')
    buffer = StreamBuffer(rid, 'codec_tokens', 'Talker', connection.chunk_policy_factory())
    manager.add_request(rid, {})
    parts = []
    def drain():
        while buffer.has_chunk_ready():
            chunk = buffer.pop_chunk()
            data = chunk.data['data']
            if data is None:
                data = torch.empty((0, model.config.num_code_groups), dtype=torch.long)
            batch = _batch(manager.engine, 'Code2Wav', 'code2wav_chunk',
                {rid: {'codec_tokens': [data.to(device)]}},
                metadata={rid: InputMetadata(stream_chunks={'codec_tokens': chunk.info})},
                final=(rid,) if chunk.is_final else ())
            manager.engine.reserve_replay_slot(batch)
            result = manager.engine.exec_and_postprocess(batch)
            if batch.admit_error is not None or batch.failed_requests:
                raise RuntimeError('speech vocoder execution failed')
            parts.append(result.per_rid_outputs[rid]['audio_chunk'][0].detach().cpu().reshape(-1))
    for index, frame in enumerate(frames):
        key = str(index)
        buffer.pre_read_register(key)
        buffer.put(key, frame)
        drain()
    buffer.signal_done()
    drain()
    manager.remove_request(rid)
    return torch.cat(parts)


@torch.no_grad()
def continuations(adapter, name, acceleration, manager, static, details):
    cases, judge_assets = adapter.speech_fixtures
    directory = Path(tempfile.mkdtemp(prefix='mstar-speech-qualification-'))
    record = {'policy': 'paired-speech-content-v1', 'artifacts': str(directory),
              'batches': list(BATCHES), 'max_frames': MAX_FRAMES,
              'conditioning': 'shared pure-text logical embeddings; own codec/state feedback',
              'execution': 'both Talkers and shared Code2Wav execute through M*',
              'fixtures': [{k: v for k, v in case.items() if not isinstance(v, torch.Tensor)} for case in cases],
              'paths': {}}
    details['speech_continuations'] = record
    generated = []
    for batch in BATCHES:
        groups = [[case] for case in cases] if batch == 1 else [[cases[i % len(cases)] for i in range(batch)]]
        for group in groups:
            print(f'Talker speech: candidate batch={batch}', flush=True)
            for row, (case, codes, stop) in enumerate(_candidate(adapter, name, acceleration, manager, group, static, record['paths'])):
                generated.append((f'b{batch}/{case["id"]}/{row}', case, codes, stop))
    references = {}
    for case in cases:
        print(f'Talker speech: reference {case["id"]}', flush=True)
        references[case['id']] = _reference(adapter, case)
    decoder = EngineManager()
    adapter.session.managers.append(decoder)
    decoder.load({'Code2Wav'}, adapter.device, {'max_concurrent_requests': 1, 'autocast_dtype': str(adapter.dtype).removeprefix('torch.')},
                 WorkerParallelGroups(1, 0), None, adapter.model, None)
    # Compilation on first execution is the ordinary Code2Wav behavior. There
    # is no full-service startup or unrelated capture-bucket warmup campaign.
    waveforms = []
    for key, case, codes, stop in generated:
        torch.save(codes, directory / (key.replace('/', '-') + '.pt'))
        waveforms.append((key, case, stop, _decode(decoder, adapter.model, adapter.device, codes, 600)))
    reference_waveforms = {case['id']: (_decode(decoder, adapter.model, adapter.device, references[case['id']][0], 600),
                                       references[case['id']][1]) for case in cases}
    judge = SpeechJudge(judge_assets)
    sr = adapter.model.get_output_sample_rate('audio')
    reference_outputs = {case['id']: judge.observe(case['text'], reference_waveforms[case['id']][0], sr,
        reference_waveforms[case['id']][1], directory / f'reference-{case["id"]}.wav') for case in cases}
    checks = []
    for key, case, stop, pcm in waveforms:
        actual = judge.observe(case['text'], pcm, sr, stop, directory / (key.replace('/', '-') + '.wav'))
        checks.append(Comparison('speech/' + key, actual, reference_outputs[case['id']]))
    return checks
