"""ASR backend interface plus the fake, MLX and CUDA implementations.

The heavy runtimes are imported inside __init__ so this module imports with
neither mlx-audio nor qwen_asr installed.
"""
from __future__ import annotations

import logging
import os
import tempfile
import wave
from typing import Protocol, runtime_checkable

import numpy as np

_SAMPLE_RATE = 16_000
log = logging.getLogger(__name__)


@runtime_checkable
class ASRBackend(Protocol):
    name: str

    def transcribe(self, chunks: list[np.ndarray]) -> list[str]:
        """One string per chunk, same length and order as `chunks`."""
        ...


class FakeBackend:
    """No model, no dependencies. Returns a placeholder per chunk."""

    name = "fake"

    def transcribe(self, chunks: list[np.ndarray]) -> list[str]:
        return [f"[chunk {i}: {len(c) / _SAMPLE_RATE:.1f} s]" for i, c in enumerate(chunks)]


def _write_wav(path: str, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


class MlxBackend:
    """MoulSot through mlx-audio on Apple silicon."""

    name = "mlx"

    def __init__(self, model_path: str):
        try:
            from mlx_audio.stt.utils import load  # type: ignore
        except ImportError:
            from mlx_audio.stt.utils import load_model as load  # type: ignore
        log.info("loading MLX model %s", model_path)
        self._model = load(model_path)
        self._accepts_array: bool | None = None  # discovered on first call

    def _generate_text(self, chunk: np.ndarray) -> str:
        if self._accepts_array is not False:
            try:
                text = self._model.generate(np.ascontiguousarray(chunk, dtype=np.float32)).text
                self._accepts_array = True
                return text
            except (TypeError, ValueError, AttributeError) as exc:
                if self._accepts_array is True:
                    raise
                log.info("generate() rejected an in-memory array (%s); using temp wav files", exc)
                self._accepts_array = False
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            _write_wav(path, chunk)
            return self._model.generate(path).text
        finally:
            os.unlink(path)

    def transcribe(self, chunks: list[np.ndarray]) -> list[str]:
        return [self._generate_text(c) for c in chunks]


class CudaBackend:
    """MoulSot through qwen_asr on a CUDA GPU. Written here, run on the GPU machine."""

    name = "cuda"

    def __init__(self, model_path: str, device: str = "cuda"):
        import torch  # type: ignore
        from qwen_asr import Qwen3ASRModel  # type: ignore

        log.info("loading CUDA model %s on %s", model_path, device)
        self._model = Qwen3ASRModel.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device
        )

    def transcribe(self, chunks: list[np.ndarray]) -> list[str]:
        if not chunks:
            return []
        audio = [(np.ascontiguousarray(c, dtype=np.float32), _SAMPLE_RATE) for c in chunks]
        results = self._model.transcribe(audio=audio, language=["Arabic"] * len(audio))
        return [r.text for r in results]


def load_backend(settings) -> ASRBackend:
    if settings.asr_backend == "fake":
        return FakeBackend()
    if settings.asr_backend == "mlx":
        return MlxBackend(settings.model_path)
    if settings.asr_backend == "cuda":
        return CudaBackend(settings.model_path, settings.model_device)
    raise ValueError(f"unknown ASR backend {settings.asr_backend!r}")
