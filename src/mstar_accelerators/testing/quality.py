"""Central consumers of deployment fidelity and model-declared quality anchors.

No acceleration supplies a metric, threshold, fixture or scoring callback.
The first metric is normalized English word error rate; additional metrics must
be explicit platform policy changes, not arbitrary callables in configuration.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

from mstar.model.fidelity import resolve_fidelity
from .assets import verify
from .harness import Unavailable
from .numerics import compare, differential_comparator
from .text import word_edits


@dataclass(frozen=True)
class QualityOutput:
    samples: tuple[dict, ...]


def resolve_quality(model, deployment, identity):
    fidelity = resolve_fidelity(deployment)
    if fidelity.mode == "numerical":
        return None
    baselines = model.quality_baselines()
    if fidelity.at_least not in baselines:
        raise ValueError(f"unknown model quality baseline: {fidelity.at_least}")
    spec = baselines[fidelity.at_least]
    if (spec['model'], spec['revision']) != (identity.get('checkpoint'), identity.get('revision')):
        raise ValueError("quality baseline does not apply to this model/checkpoint")
    if spec['metric'] != 'english_word_error_rate_v1':
        raise Unavailable('quality_metric', 'no central consumer for this model metric')
    if spec['normalizer_revision'] != '86098128c0b4f24f0e2aa2994de830614b474227':
        raise ValueError("quality metric normalizer differs from pinned consumer")
    if not spec['samples'] or len({s['id'] for s in spec['samples']}) != len(spec['samples']):
        raise ValueError("quality corpus must contain distinct samples")
    for sample in spec['samples']:
        fixture_path(sample)
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return {**spec, 'name': fidelity.at_least, 'descriptor_sha256': digest}


def fixture_path(sample):
    name = sample['path']
    if Path(name).name != name:
        raise ValueError("quality fixture must be a corpus filename")
    path = Path(__file__).with_name('corpus_data') / 'quality' / name
    if not path.is_file():
        raise Unavailable('quality_fixture', f'missing pinned quality fixture {name}')
    verify(path, {'bytes': sample['bytes'], 'sha256': sample['audio_sha256']})
    return path


def score(name, output, spec):
    from ._vendor.english_normalizer import EnglishTextNormalizer
    normalizer = EnglishTextNormalizer()
    result = {'name': name, 'passed': False, 'policy': spec['metric'],
              'baseline': spec['name'], 'descriptor_sha256': spec['descriptor_sha256']}
    if not isinstance(output, QualityOutput):
        return result | {'reason': 'missing task quality outputs'}
    if [s.get('id') for s in output.samples] != [s['id'] for s in spec['samples']]:
        return result | {'reason': 'missing, reordered or duplicated quality samples'}
    rows, edits, words = [], 0, 0
    for sample, actual in zip(spec['samples'], output.samples):
        ref = normalizer(sample['reference'])
        if ref != sample['normalized_reference']:
            raise ValueError('normalizer disagrees with frozen baseline labels')
        hyp = normalizer(actual['text'])
        n = word_edits(ref.split(), hyp.split())
        edits += n
        words += len(ref.split())
        rows.append({**actual, 'normalized_hypothesis': hyp, 'word_edits': n,
                     'reference_words': len(ref.split())})
    if words != spec['baseline']['reference_words']:
        raise ValueError('baseline denominator disagrees with evaluation labels')
    passed = edits <= spec['baseline']['word_edits'] and all(s['stop'] == 'eos' for s in rows)
    return result | {'passed': passed, 'reason': 'fixed task-quality requirement',
                     'word_edits': edits, 'reference_words': words, 'wer': edits / words,
                     'maximum_word_edits': spec['baseline']['word_edits'], 'samples': rows}


def quality_comparator(acceleration, observation):
    if isinstance(observation.actual, QualityOutput):
        return score(observation.name, observation.actual, observation.expected)
    import torch
    if (not observation.exact and isinstance(observation.actual, torch.Tensor)
            and observation.actual.is_floating_point()):
        return compare(observation.name, observation.actual, observation.expected,
                       dtype=observation.dtype, suppression_mask=observation.suppression_mask,
                       structural_only=True, internal=observation.internal)
    return differential_comparator(acceleration, observation)
