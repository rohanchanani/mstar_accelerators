#!/usr/bin/env python3
"""Exercise M*'s retained AudioDecoder call and compare a PCM16 fixture exactly."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
import wave


def verify(audio, process):
    import torch
    import torchcodec
    from torchcodec.decoders import AudioDecoder

    with wave.open(str(audio), "rb") as source:
        assert source.getframerate() == 16000 and source.getnchannels() == 1
        assert source.getsampwidth() == 2
        expected = torch.frombuffer(bytearray(source.readframes(source.getnframes())), dtype=torch.int16).float() / 32768
    # Exactly the retained Model.load_audio primitive and result projection.
    decoder = AudioDecoder(str(audio), sample_rate=16000, num_channels=1)
    actual = decoder.get_all_samples().data[0]
    assert torch.equal(actual, expected), (actual.shape, expected.shape, (actual - expected).abs().max())
    libraries = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                 if any(x in line for x in ("libtorchcodec", "libavcodec", "libavformat", "libavutil", "libswresample"))}
    print(json.dumps({"status": "PASS", "process": process, "torch": torch.__version__,
                      "torchcodec": torchcodec.__version__, "samples": actual.numel(),
                      "sample_rate": 16000, "channels": 1,
                      "pcm_f32_sha256": hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
                      "loaded_libraries": sorted(libraries)}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--spawn", action="store_true")
    args = parser.parse_args()
    verify(args.audio, "parent")
    if args.spawn:
        child = multiprocessing.get_context("spawn").Process(target=verify, args=(args.audio, "spawn"))
        child.start()
        child.join()
        if child.exitcode != 0:
            raise RuntimeError(f"TorchCodec spawned check failed: {child.exitcode}")


if __name__ == "__main__":
    main()
