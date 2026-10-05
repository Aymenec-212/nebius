"""Runs decode -> VAD -> chunk -> transcribe -> assemble. Knows nothing about FastAPI
or which ASR backend is active."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from app.asr import ASRBackend
from app.audio import SAMPLE_RATE, AudioTooLongError, decode_audio
from app.config import Settings
from app.vad import Span, detect_speech, fixed_windows, make_chunks


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    segments: list[Segment]
    audio_duration_s: float
    processing_s: float
    backend: str

    def to_dict(self) -> dict:
        return asdict(self)


def _slice(audio: np.ndarray, span: Span) -> np.ndarray:
    start = max(0, int(round(span.start * SAMPLE_RATE)))
    end = min(audio.shape[0], int(round(span.end * SAMPLE_RATE)))
    return audio[start:end]


def transcribe_bytes(
    data: bytes, backend: ASRBackend, vad, settings: Settings, use_vad: bool = True
) -> TranscriptionResult:
    t0 = time.perf_counter()
    audio = decode_audio(data)
    total_s = audio.shape[0] / SAMPLE_RATE
    if total_s > settings.max_audio_s:
        raise AudioTooLongError(
            f"audio is {total_s:.1f} s, limit is {settings.max_audio_s:.0f} s"
        )

    if use_vad:
        spans = detect_speech(audio, vad)
        chunks = make_chunks(
            spans, total_s, settings.max_chunk_s, settings.merge_gap_s, settings.pad_s
        )
    else:
        chunks = fixed_windows(total_s, settings.max_chunk_s)

    segments: list[Segment] = []
    if chunks:
        arrays = [_slice(audio, c) for c in chunks]
        texts = backend.transcribe(arrays)
        if len(texts) != len(chunks):
            raise RuntimeError(
                f"backend returned {len(texts)} texts for {len(chunks)} chunks"
            )
        segments = [
            Segment(round(c.start, 3), round(c.end, 3), t.strip())
            for c, t in zip(chunks, texts)
        ]

    text = " ".join(s.text for s in segments if s.text)
    return TranscriptionResult(
        text=text,
        segments=segments,
        audio_duration_s=round(total_s, 3),
        processing_s=round(time.perf_counter() - t0, 3),
        backend=backend.name,
    )
