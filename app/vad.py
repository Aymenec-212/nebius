"""Speech detection (Silero VAD) and chunk arithmetic. Knows nothing about ASR or HTTP."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Span:
    start: float  # seconds
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def load_vad():
    """Load the Silero VAD model once at startup (bundled with the silero-vad package)."""
    from silero_vad import load_silero_vad

    return load_silero_vad()


def detect_speech(audio: np.ndarray, vad) -> list[Span]:
    """Return speech spans in seconds for a 16 kHz mono float32 array."""
    import torch
    from silero_vad import get_speech_timestamps

    if audio.size == 0:
        return []
    tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    stamps = get_speech_timestamps(tensor, vad, sampling_rate=_SAMPLE_RATE, return_seconds=True)
    return [Span(float(s["start"]), float(s["end"])) for s in stamps]


def make_chunks(
    spans: list[Span], total_s: float, max_chunk_s: float, merge_gap_s: float, pad_s: float
) -> list[Span]:
    """Merge close spans, split long ones, pad and clamp. Pure arithmetic.

    1. Merge consecutive spans when the gap is <= merge_gap_s and the merged span
       stays <= max_chunk_s.
    2. Split any span longer than max_chunk_s into equal parts <= max_chunk_s.
    3. Pad each chunk by pad_s on both sides and clamp to [0, total_s].
    4. Return sorted by start. Empty in, empty out.
    """
    if max_chunk_s <= 0:
        raise ValueError("max_chunk_s must be positive")
    ordered = sorted((s for s in spans if s.end > s.start), key=lambda s: s.start)
    if not ordered:
        return []

    merged: list[Span] = [ordered[0]]
    for span in ordered[1:]:
        last = merged[-1]
        gap = span.start - last.end
        if gap <= merge_gap_s and max(span.end, last.end) - last.start <= max_chunk_s:
            merged[-1] = Span(last.start, max(last.end, span.end))
        else:
            merged.append(span)

    split: list[Span] = []
    for span in merged:
        if span.duration <= max_chunk_s:
            split.append(span)
            continue
        parts = math.ceil(span.duration / max_chunk_s)
        step = span.duration / parts
        for i in range(parts):
            start = span.start + i * step
            end = span.end if i == parts - 1 else start + step
            split.append(Span(start, end))

    padded = [
        Span(max(0.0, s.start - pad_s), min(total_s, s.end + pad_s)) for s in split
    ]
    return sorted((s for s in padded if s.end > s.start), key=lambda s: s.start)


def fixed_windows(total_s: float, max_chunk_s: float) -> list[Span]:
    """Cut [0, total_s] into consecutive windows of at most max_chunk_s (used when vad=false)."""
    if max_chunk_s <= 0:
        raise ValueError("max_chunk_s must be positive")
    if total_s <= 0:
        return []
    windows: list[Span] = []
    start = 0.0
    while start < total_s:
        end = min(total_s, start + max_chunk_s)
        windows.append(Span(start, end))
        start = end
    return windows
