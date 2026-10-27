from dataclasses import replace
from mstar_accelerators.testing.speech import SpeechOutput, score_speech

TEXT = 'Use bullet points sparingly and keep each slide focused on one key idea.'


def clip(text=TEXT):
    return SpeechOutput(TEXT, text, 'eos', True, 5., 4.)


def test_clean_pair_passes_but_one_bad_clip_cannot_hide_in_aggregate():
    assert score_speech('clean', clip(), clip())['passed']
    assert not score_speech('wrong-speech', clip('Something completely unrelated.'), clip())['passed']
    assert not score_speech('extra-opening', clip('Bulltar ' + TEXT), clip())['passed']


def test_opening_detects_a_small_error_that_whole_clip_wer_can_hide():
    text = TEXT + ' ' + 'This is a much longer continuation. ' * 10
    ref = replace(clip(text), text=text)
    bad = replace(ref, transcript='Bulltar ' + text)
    result = score_speech('opening', bad, ref)
    assert result['candidate']['wer'] < .05
    assert not result['passed']


def test_reference_failure_is_not_candidate_success():
    assert not score_speech('bad-reference', clip(), clip(''))['passed']


def test_truncation_silence_and_invalid_payload_are_rejected():
    for bad in (replace(clip(), stop='limit'), replace(clip(), voiced_s=0.),
                replace(clip(), finite=False), replace(clip(), duration_s=float('nan'))):
        assert not score_speech('invalid', bad, clip())['passed']
