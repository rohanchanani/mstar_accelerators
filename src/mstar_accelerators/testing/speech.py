"""Ordinary speech-generation checks, owned by the qualification platform.

Every clip must preserve its conditioning text at least as well as its paired
reference, including the opening sentence. ASR word errors follow the
Qwen/vLLM-Omni speech validation precedent; paired non-regression and separate
opening checks are explicit platform policy, not an upstream threshold port.
These checks are bounded content evidence,
not a perceptual-quality proof or a replacement for state/numerical checks.
"""
from dataclasses import dataclass
from pathlib import Path
import math
import re
import torch
from .text import word_edits
from ._vendor.english_normalizer import EnglishTextNormalizer
from .harness import Unavailable

POLICY = 'paired-speech-content-v1'
JUDGE_ID = 'openai/whisper-large-v3-turbo'
JUDGE_REVISION = '41f01f3fe87f28c78e2fbf8b568835947dd65ed9'


@dataclass(frozen=True)
class SpeechOutput:
    text: str
    transcript: str
    stop: str
    finite: bool
    duration_s: float
    voiced_s: float
    wav: str = ''


def score_speech(name, actual, expected):
    result = {'name': name, 'passed': False, 'policy': POLICY,
              'judge': {'model': JUDGE_ID, 'revision': JUDGE_REVISION}}
    if not isinstance(actual, SpeechOutput) or not isinstance(expected, SpeechOutput):
        return result | {'reason': 'missing paired speech observation'}
    if actual.text != expected.text:
        return result | {'reason': 'different conditioning text'}
    normalizer = EnglishTextNormalizer()
    target = normalizer(expected.text).split()
    if not target:
        return result | {'reason': 'empty conditioning text'}
    opening = normalizer(re.split(r'[.!?](?:\s|$)', expected.text, maxsplit=1)[0]).split()
    result['opening_words'] = len(opening)
    rows = {}
    for label, item in (('candidate', actual), ('reference', expected)):
        heard = normalizer(item.transcript).split()
        edits = word_edits(target, heard)
        n = len(opening)
        rows[label] = {'transcript': item.transcript, 'wav': item.wav,
            'word_edits': edits, 'words': len(target), 'wer': edits / len(target),
            'opening_word_edits': word_edits(opening, heard[:n]), 'stop': item.stop,
            'duration_s': item.duration_s if math.isfinite(item.duration_s) else None,
            'voiced_s': item.voiced_s if math.isfinite(item.voiced_s) else None,
            'nonempty_transcript': bool(heard),
            'signal_valid': item.finite and math.isfinite(item.duration_s)
                and math.isfinite(item.voiced_s) and item.duration_s >= item.voiced_s >= 0.3}
    def valid(row):
        return row['signal_valid'] and row['stop'] == 'eos' and row['nonempty_transcript']
    if not valid(rows['reference']):
        return result | {'reason': 'reference continuation failed the fixed speech requirement', **rows}
    passed = (valid(rows['candidate'])
              and rows['candidate']['word_edits'] <= rows['reference']['word_edits']
              and rows['candidate']['opening_word_edits'] <= rows['reference']['opening_word_edits'])
    return result | {'passed': passed, 'reason': 'per-clip speech and opening comparison', **rows}


def judge_path():
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        path = Path(snapshot_download(JUDGE_ID, revision=JUDGE_REVISION,
            allow_patterns=['*.json', '*.txt', '*.safetensors'], local_files_only=True))
    except LocalEntryNotFoundError as exc:
        raise Unavailable('speech_judge_assets', f'missing pinned speech judge {JUDGE_ID}@{JUDGE_REVISION}') from exc
    if not (path / 'model.safetensors').is_file():
        raise Unavailable('speech_judge_assets', 'pinned speech judge weights are missing')
    return path


class SpeechJudge:
    """One CPU ASR evaluator shared by all clips, outside inference timing."""
    def __init__(self, path):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.processor = WhisperProcessor.from_pretrained(path, local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.float32, local_files_only=True).eval()

    @torch.inference_mode()
    def observe(self, text, pcm, sample_rate, stop, path):
        import wave
        from scipy.signal import resample_poly
        pcm = pcm.detach().cpu().reshape(-1)
        if pcm.dtype != torch.int16:
            raise ValueError('speech continuation must return the production PCM16 interface')
        path = Path(path)
        with wave.open(str(path), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.numpy().astype('<i2').tobytes())
        signal = pcm.float().numpy() / 32768.
        finite = bool(torch.isfinite(pcm.float()).all())
        # Match Atindra's frame RMS activity check without removing silence
        # from the waveform that ASR receives.
        width = max(1, sample_rate // 50)
        framed = torch.from_numpy(signal[:len(signal)//width*width]).reshape(-1, width)
        voiced = float((framed.square().mean(1).sqrt() > 0.01).sum()) * width / sample_rate
        heard = ''
        if len(signal) and finite:
            divisor = math.gcd(sample_rate, 16000)
            audio = resample_poly(signal, 16000//divisor, sample_rate//divisor)
            long = len(audio) > 30*16000
            inputs = self.processor(audio, sampling_rate=16000, return_tensors='pt',
                truncation=not long, padding='longest' if long else 'max_length',
                return_attention_mask=True)
            ids = self.model.generate(inputs.input_features, attention_mask=inputs.attention_mask,
                language='en', task='transcribe', do_sample=False,
                return_timestamps=long, max_new_tokens=440)
            heard = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return SpeechOutput(text, heard, stop, finite, len(signal)/sample_rate, voiced, str(path))
