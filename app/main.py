"""HTTP layer: routes, request parsing, error -> status, model loading, inference lock."""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.asr import load_backend
from app.audio import AudioDecodeError, AudioTooLongError
from app.config import load_settings
from app.pipeline import transcribe_bytes
from app.vad import load_vad

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("moulsot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.settings = load_settings()
    app.state.lock = threading.Lock()
    app.state.vad = await run_in_threadpool(load_vad)
    app.state.backend = await run_in_threadpool(load_backend, app.state.settings)
    app.state.ready = True
    log.info("ready with backend=%s", app.state.backend.name)
    yield
    app.state.ready = False


app = FastAPI(title="MoulSot API", lifespan=lifespan)


def _error(status: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": detail})


@app.get("/health")
async def health(request: Request):
    if not getattr(request.app.state, "ready", False):
        return _error(503, "loading")
    return {"status": "ok", "backend": request.app.state.backend.name}


def _run_locked(state, data: bytes, use_vad: bool):
    with state.lock:
        return transcribe_bytes(data, state.backend, state.vad, state.settings, use_vad=use_vad)


@app.post("/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile | None = File(default=None),
    vad: bool = Query(default=True),
):
    state = request.app.state
    if not getattr(state, "ready", False):
        return _error(503, "loading")
    if file is None:
        return _error(400, "multipart field 'file' is required")

    limit = state.settings.max_upload_bytes
    data = await file.read(limit + 1)
    if not data:
        return _error(400, "uploaded file is empty")
    if len(data) > limit:
        return _error(413, f"upload exceeds {state.settings.max_upload_mb:g} MB")

    try:
        result = await run_in_threadpool(_run_locked, state, data, vad)
    except AudioDecodeError as exc:
        return _error(400, f"could not decode audio: {exc}")
    except AudioTooLongError as exc:
        return _error(413, str(exc))
    except Exception:
        log.exception("transcription failed")
        return _error(500, "transcription failed")
    return result.to_dict()
