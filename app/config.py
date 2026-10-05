"""Settings read once from environment variables. Depends on nothing else."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

BACKENDS = ("fake", "mlx", "cuda")


@dataclass(frozen=True)
class Settings:
    asr_backend: str = "fake"
    model_path: str = "atlasia/moulsot.v0.3"
    model_device: str = "cuda"
    max_upload_mb: float = 25.0
    max_audio_s: float = 600.0
    max_chunk_s: float = 20.0
    merge_gap_s: float = 0.5
    pad_s: float = 0.2
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build a Settings from `env` (defaults to os.environ)."""
    e = os.environ if env is None else env
    backend = e.get("ASR_BACKEND", Settings.asr_backend).strip().lower()
    if backend not in BACKENDS:
        raise ValueError(f"ASR_BACKEND must be one of {BACKENDS}, got {backend!r}")
    return Settings(
        asr_backend=backend,
        model_path=e.get("MODEL_PATH", Settings.model_path),
        model_device=e.get("MODEL_DEVICE", Settings.model_device),
        max_upload_mb=float(e.get("MAX_UPLOAD_MB", Settings.max_upload_mb)),
        max_audio_s=float(e.get("MAX_AUDIO_S", Settings.max_audio_s)),
        max_chunk_s=float(e.get("MAX_CHUNK_S", Settings.max_chunk_s)),
        merge_gap_s=float(e.get("MERGE_GAP_S", Settings.merge_gap_s)),
        pad_s=float(e.get("PAD_S", Settings.pad_s)),
        host=e.get("HOST", Settings.host),
        port=int(e.get("PORT", Settings.port)),
    )
