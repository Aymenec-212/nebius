"""Bytes -> 16 kHz mono float32 array via ffmpeg. Knows nothing about VAD, ASR or HTTP."""
from __future__ import annotations

import subprocess

import numpy as np

SAMPLE_RATE = 16_000

_FFMPEG_CMD = [
    "ffmpeg",
    "-nostdin",
    "-hide_banner",
    "-loglevel", "error",
    "-i", "pipe:0",
    "-vn",
    "-f", "f32le",
    "-acodec", "pcm_f32le",
    "-ac", "1",
    "-ar", str(SAMPLE_RATE),
    "pipe:1",
]


class AudioDecodeError(Exception):
    """The input is empty or ffmpeg could not decode it."""


class AudioTooLongError(Exception):
    """The decoded audio is longer than the configured limit."""


def decode_audio(data: bytes) -> np.ndarray:
    """Decode any ffmpeg-readable container/codec to float32 mono 16 kHz in [-1, 1].

    ffmpeg reads from stdin and writes raw PCM to stdout; no temp files are used.
    """
    if not data:
        raise AudioDecodeError("empty input")
    try:
        proc = subprocess.run(_FFMPEG_CMD, input=data, capture_output=True, check=False)
    except FileNotFoundError as exc:  # pragma: no cover - environment problem
        raise AudioDecodeError("ffmpeg executable not found") from exc
    if proc.returncode != 0 or not proc.stdout:
        msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise AudioDecodeError(msg[-1] if msg else "ffmpeg produced no audio")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise AudioDecodeError("decoded audio is empty")
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)
