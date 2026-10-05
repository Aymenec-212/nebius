import os
import subprocess
from pathlib import Path

import pytest

os.environ.setdefault("ASR_BACKEND", "fake")

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
SHORT_SAMPLE = SAMPLES / "short.ogg"
LONG_SAMPLE = SAMPLES / "long.ogg"


def ffmpeg_synth(path: Path, source: str, duration_s: float, rate: int = 16000, channels: int = 1,
                 codec: list[str] | None = None) -> bytes:
    """Generate a fixture with ffmpeg's lavfi sources (sine, anullsrc, ...)."""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"{source}:duration={duration_s}",
        "-ar", str(rate), "-ac", str(channels),
    ] + (codec or []) + [str(path)]
    subprocess.run(cmd, check=True)
    return path.read_bytes()


@pytest.fixture(scope="session")
def tone_wav(tmp_path_factory) -> bytes:
    p = tmp_path_factory.mktemp("audio") / "tone.wav"
    return ffmpeg_synth(p, "sine=frequency=440:sample_rate=16000", 2.0)


@pytest.fixture(scope="session")
def silence_wav(tmp_path_factory) -> bytes:
    p = tmp_path_factory.mktemp("audio") / "silence.wav"
    return ffmpeg_synth(p, "anullsrc=r=16000:cl=mono", 3.0)


@pytest.fixture(scope="session")
def short_sample() -> bytes:
    if not SHORT_SAMPLE.exists():
        pytest.skip(f"{SHORT_SAMPLE} not present; add a ~10 s Darija ogg/opus sample")
    return SHORT_SAMPLE.read_bytes()
