"""Runs the real MoulSot model through mlx-audio. Skipped when mlx_audio is not installed."""
import re

import pytest

pytest.importorskip("mlx_audio")

from app.asr import MlxBackend  # noqa: E402
from app.config import Settings  # noqa: E402
from app.pipeline import transcribe_bytes  # noqa: E402
from app.vad import load_vad  # noqa: E402

ARABIC = re.compile(r"[؀-ۿ]")


def test_short_sample_is_arabic_script(short_sample):
    settings = Settings(asr_backend="mlx")
    backend = MlxBackend(settings.model_path)
    r = transcribe_bytes(short_sample, backend, load_vad(), settings)
    assert r.backend == "mlx"
    assert len(r.segments) >= 1
    for s in r.segments:
        assert s.text, f"empty text for segment {s.start}-{s.end}"
        assert ARABIC.search(s.text), f"no Arabic script in {s.text!r}"
    assert ARABIC.search(r.text)
