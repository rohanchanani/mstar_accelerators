# Fixed Whisper demo assets

`assets.json` records the complete pinned Hugging Face checkpoint file list and
SHA-256 identities. The checkpoint is fetched outside source by
`tools/whisper/prepare_assets.py`; model weights are not shipped in this repo.

`jfk16k.wav` is the exact 16 kHz mono PCM16 conversion of OpenAI Whisper's
`tests/jfk.flac` at revision `6e3be77e1a105e59086e3e21ff5f609fd6fa89a5`.
The source URL/hash, conversion tool and command are recorded in `assets.json`.
`jfk_tail16k.wav` takes frames 80,000 through 175,999 from that WAV without
another resampling operation. These short fixtures are distinct inputs for
request-state isolation; their frozen bytes avoid decoder/resampler variability.

Original source and license:

- https://github.com/openai/whisper/tree/6e3be77e1a105e59086e3e21ff5f609fd6fa89a5/tests
- https://github.com/openai/whisper/blob/6e3be77e1a105e59086e3e21ff5f609fd6fa89a5/LICENSE

`reference.json` records tokens, EOS, positions, sampling offsets and text from
the original HF encoder followed by the original M* decoder on the pinned Tiny
checkpoint. The full recording produces 24 tokens including EOS; the tail
produces 17. The tail's word **as** is the model's observed output, not an edited
transcription. The complete generation reports are included in the delivery
evidence. Do not rewrite the oracle to agree with a candidate.
