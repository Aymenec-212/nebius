# MoulSot API — local stage

A small FastAPI service that turns a Moroccan Darija audio file into text with
timestamps, using [atlasia/moulsot.v0.3](https://huggingface.co/atlasia/moulsot.v0.3).

One request runs six stages: receive → decode (ffmpeg) → detect speech (Silero VAD)
→ chunk → transcribe → assemble. Only the transcribe stage differs between machines,
behind one `ASRBackend` interface with three implementations:

| Backend | Runtime | Use |
| --- | --- | --- |
| `fake` | none | tests, startup checks (default) |
| `mlx` | `mlx-audio` on Apple silicon | real local transcription |
| `cuda` | `qwen_asr` on an NVIDIA GPU | the GPU stage (written, not yet run) |

## Run locally on an M1 Mac

```bash
brew install ffmpeg
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements-mac.txt      # shared deps + mlx-audio

ASR_BACKEND=mlx python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# or, without activating the venv:
ASR_BACKEND=mlx uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`uv venv` is used instead of `python -m venv` because the latter can fail in
`ensurepip` on Homebrew Pythons. Launch with `python -m uvicorn` or `uv run uvicorn`
rather than a bare `uvicorn`: a `uvicorn` executable from another Python on your
`PATH` will not see the dependencies installed in `.venv` and fails with
`ModuleNotFoundError: No module named 'fastapi'`.

The first `mlx` start downloads the weights from Hugging Face; later starts load
them from the local cache. `GET /health` returns `503` until loading is done.

```bash
curl localhost:8000/health
# {"status":"ok","backend":"mlx"}

curl -F file=@samples/long.ogg localhost:8000/transcribe
curl -F file=@samples/long.ogg "localhost:8000/transcribe?vad=false"   # fixed windows
```

Response:

```json
{
  "text": "...",
  "segments": [{"start": 0.4, "end": 7.9, "text": "..."}],
  "audio_duration_s": 312.5,
  "processing_s": 1.8,
  "backend": "mlx"
}
```

| Status | When |
| --- | --- |
| 400 | `file` missing, empty, or not decodable |
| 413 | upload over `MAX_UPLOAD_MB`, or audio longer than `MAX_AUDIO_S` |
| 500 | anything else (traceback is logged, not returned) |

Transcription runs in the threadpool behind one lock, so `/health` stays
responsive and the model never sees two requests at once. Run one uvicorn worker.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ASR_BACKEND` | `fake` | `fake`, `mlx` or `cuda` |
| `MODEL_PATH` | `atlasia/moulsot.v0.3` | HF id or local folder |
| `MODEL_DEVICE` | `cuda` | device for the CUDA backend |
| `MAX_UPLOAD_MB` | `25` | upload size limit |
| `MAX_AUDIO_S` | `600` | audio duration limit |
| `MAX_CHUNK_S` | `20` | longest chunk sent to the model |
| `MERGE_GAP_S` | `0.5` | largest silence merged inside one chunk |
| `PAD_S` | `0.2` | padding on each side of a chunk |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | bind address |

Each returned segment is at most `MAX_CHUNK_S + 2 × PAD_S` long.

## Tests

```bash
uv pip install -r requirements-mac.txt   # or requirements.txt on Linux
python -m pytest            # ASR_BACKEND=fake is set by tests/conftest.py
```

Put `samples/short.ogg` (~10 s) and `samples/long.ogg` (5+ min) of Darija speech in
`samples/` (they are git-ignored). Tests that need them skip when they are missing.
`tests/test_mlx.py` runs the real model and is skipped unless `mlx_audio` is installed.

## Layout

```
app/config.py    settings from env vars
app/audio.py     bytes -> 16 kHz mono float32 via ffmpeg (stdin -> stdout)
app/vad.py       Silero VAD + pure chunking arithmetic
app/asr.py       ASRBackend protocol; Fake, Mlx and Cuda backends (lazy imports)
app/pipeline.py  runs the stages and builds the result
app/main.py      FastAPI routes, error mapping, startup loading, inference lock
```

Dependencies point downward only: `main` → `pipeline` → components. Components never
import each other, and nothing outside `main.py` imports FastAPI.
