# Presentation results

The README preserves the two October 25, 2026 results. They describe measured
implementations before the Pi/AMD integration. The final merged artifact needs
its own qualification and benchmark; moving source is not a new measurement.

## Whisper-tiny / Raspberry Pi 5 CPU

Warm, complete-transcript HTTP latency on six fixed English development clips,
one request at a time, four repetitions per implementation in ABBA blocks.
The headline is the mean of the six per-clip medians, not the fastest request.
Both services use multilingual Whisper-tiny, full encoder context and greedy
decoding through the same HTTP boundary, with weights already loaded.

| Metric | Optimized whisper.cpp + KleidiAI | Accelerated M* |
| --- | ---: | ---: |
| Mean of per-clip medians | 460.2687 ms | 325.4690 ms |
| Mean of all 24 measured requests | 465.2962 ms | 326.4311 ms |
| Normalized word error rate | 10/87 = 11.5% | 7/87 = 8.0% |

Latency reduction: `100 × (1 − 325.468956 / 460.268658) = 29.3%`.
The public baseline uses Q8_0, four threads and Flash Attention. The two systems
can generate different token sequences; this is not identical-work kernel timing.
The six development clips do not establish generalization to held-out audio.
The V3D implementation is a separate supported route, not the source of this result.

Source receipt: `PUBLIC_CROSS_Q1_REFRESH.json`.

- M* source: `5c4d00880479b641f42b6f4ef56f3ba088812a36`.
- External physical source: `47af5b4dc5910642a6b6e086ad054b69750562ce`;
  external runtime revision recorded as `e7b9196`.
- Native library SHA256: `c7d73d3320332fcd737381010123fa7066b657bb7c45be000c653549ff123570`.
- Public binary SHA256: `674512e71c28d59a471b8f1fbb4692c021fca81b5d0379adecc5358036506c22`.
- Dataset manifest SHA256: `512a94f30c8f493cf3e4d458f6edbb240c630f8cd042c62c3a6a687ee68e5134`.

## Qwen3-Omni / AMD Instinct MI300X

The complete five-node service runs on one MI300X VM, BF16, TP1/SP1. The unchanged
Atindra `benchmark.amd.sweep.run_cell` / `benchmark.runner` and original Omni
text/image/audio-to-speech workloads supply requests and metrics. Client
concurrency is 1, 4, 16 or 32; both deployments admit at most eight active requests,
so higher client concurrency includes queueing.

Each cell takes the median of three repetitions, using the original 5% spread
retry policy. The README percentages are calculated from unrounded medians:

- TTFA change: `100 × (accelerated ttfa_p50_s / baseline ttfa_p50_s − 1)`.
- Throughput change: `100 × (accelerated audio_s_per_s / baseline audio_s_per_s − 1)`.
- Throughput counts output audio seconds per wall second, not requests per second.

The baseline was collected earlier on the unchanged branch; it is not a fresh
interleaved comparison. The final accelerated run resumed after a service restart.
Audio c4 throughput remained flagged noisy after the benchmark's retry. Repeated
measurements within a service instance do not establish across-restart repeatability;
no table entry is labeled a repeatability-qualified best result.

| Role | Source |
| --- | --- |
| Unchanged baseline | `users/atindra/amd-native`, `678f59d6adb3d892263f2ba30fcf7b28145abf2e` |
| Measured M* | `103b715c1bfbccf0688a27a24401148bbcba5a9f` |
| Measured accelerators | `9f44ddbcb7cb92eb9b53550349d1d8ac23dd3ccd` |

Source receipts: `native-baseline.json` and `final-c-results.json`. Both retain
per-cell repetitions and retry history. The later ownership migration has separate
preservation checks; it must not be represented as a rerun of these measurements.
A speech-quality failure also reproduced in the untouched measured winner, so the
performance table is not a claim that every generated clip is intelligible.
