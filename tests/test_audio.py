import numpy as np
import pytest

from app.audio import SAMPLE_RATE, AudioDecodeError, decode_audio
from tests.conftest import ffmpeg_synth


def _check(audio: np.ndarray, duration_s: float):
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert abs(audio.shape[0] - duration_s * SAMPLE_RATE) < 0.05 * SAMPLE_RATE
    assert audio.min() >= -1.0 and audio.max() <= 1.0


def test_wav_decodes(tone_wav):
    _check(decode_audio(tone_wav), 2.0)


def test_ogg_opus_decodes(tmp_path):
    data = ffmpeg_synth(tmp_path / "tone.ogg", "sine=frequency=440:sample_rate=48000", 2.0,
                        rate=48000, codec=["-c:a", "libopus"])
    _check(decode_audio(data), 2.0)


def test_stereo_44k_is_downmixed_and_resampled(tmp_path):
    data = ffmpeg_synth(tmp_path / "stereo.wav", "sine=frequency=440:sample_rate=44100", 1.5,
                        rate=44100, channels=2)
    _check(decode_audio(data), 1.5)


def test_mp3_decodes(tmp_path):
    data = ffmpeg_synth(tmp_path / "tone.mp3", "sine=frequency=440:sample_rate=44100", 1.0,
                        rate=44100, codec=["-c:a", "libmp3lame"])
    _check(decode_audio(data), 1.0)


def test_random_bytes_raise():
    with pytest.raises(AudioDecodeError):
        decode_audio(np.random.default_rng(0).bytes(4096))


def test_empty_bytes_raise():
    with pytest.raises(AudioDecodeError):
        decode_audio(b"")
