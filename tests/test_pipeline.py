import dataclasses

import pytest

from app.asr import FakeBackend
from app.audio import AudioTooLongError
from app.config import Settings
from app.pipeline import transcribe_bytes
from app.vad import load_vad


@pytest.fixture(scope="module")
def vad():
    return load_vad()


SETTINGS = Settings()
BACKEND = FakeBackend()


def test_silence_gives_empty_result(silence_wav, vad):
    r = transcribe_bytes(silence_wav, BACKEND, vad, SETTINGS)
    assert r.text == ""
    assert r.segments == []
    assert abs(r.audio_duration_s - 3.0) < 0.05
    assert r.backend == "fake"
    assert r.processing_s >= 0


def test_short_sample_has_segments(short_sample, vad):
    r = transcribe_bytes(short_sample, BACKEND, vad, SETTINGS)
    assert len(r.segments) >= 1
    assert r.text
    last_end = 0.0
    for s in r.segments:
        assert 0.0 <= s.start < s.end <= r.audio_duration_s + 1e-6
        assert s.start >= last_end - 1e-6
        assert s.end - s.start <= SETTINGS.max_chunk_s + 2 * SETTINGS.pad_s + 1e-6
        last_end = s.start
        assert s.text.startswith("[chunk ")


def test_vad_off_uses_fixed_windows(tone_wav, vad):
    small = dataclasses.replace(SETTINGS, max_chunk_s=0.5)
    r = transcribe_bytes(tone_wav, BACKEND, vad, small, use_vad=False)
    assert len(r.segments) == 4
    assert r.segments[0].start == 0.0
    assert abs(r.segments[-1].end - r.audio_duration_s) < 1e-6
    assert r.text == " ".join(s.text for s in r.segments)


def test_overlong_audio_raises(tone_wav, vad):
    with pytest.raises(AudioTooLongError):
        transcribe_bytes(tone_wav, BACKEND, vad, dataclasses.replace(SETTINGS, max_audio_s=1.0))
