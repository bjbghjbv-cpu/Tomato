"""ShelfSense Edge - FastAPI server.

Menjalankan:
    .venv\\Scripts\\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.core.profile import list_profiles, load_profile
from backend.core.vision import Pipeline, encode_jpeg

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"

app = FastAPI(title="ShelfSense Edge", version="1.0.0")

_pipeline: Pipeline | None = None
_pipeline_lock = asyncio.Lock()


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        synthetic = os.environ.get("SSE_SYNTHETIC", "0") == "1"
        cam = int(os.environ.get("SSE_CAMERA", "0"))
        _pipeline = Pipeline(
            profile_id=os.environ.get("SSE_PROFILE", "tomato"),
            camera_index=cam,
            model_path=os.environ.get("SSE_MODEL") or None,
            synthetic=synthetic,
        )
    return _pipeline


class EnvPayload(BaseModel):
    temp_c: float | None = None
    rh: float | None = None
    time_offset_h: float | None = None


class ProfilePayload(BaseModel):
    id: str


class ScanPayload(BaseModel):
    image: str  # data URL atau base64 murni
    source: str = "auto"  # auto | color | roi


# ----------------------------------------------------------------------
# Halaman & aset
# ----------------------------------------------------------------------
@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ----------------------------------------------------------------------
# API
# ----------------------------------------------------------------------
@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    return get_pipeline().status()


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/profiles")
async def api_profiles() -> dict[str, Any]:
    out = []
    for pid in list_profiles():
        p = load_profile(pid)
        out.append(p.public_dict())
    return {"profiles": out}


@app.post("/api/env")
async def api_env(payload: EnvPayload) -> dict[str, Any]:
    pipe = get_pipeline()
    pipe.set_env(payload.temp_c, payload.rh, payload.time_offset_h)
    return {"ok": True, "config": pipe._config()}


@app.post("/api/profile")
async def api_profile(payload: ProfilePayload) -> dict[str, Any]:
    pipe = get_pipeline()
    pipe.set_profile(payload.id)
    return {"ok": True, "profile": pipe.profile.public_dict()}


@app.post("/api/calibrate")
async def api_calibrate() -> dict[str, Any]:
    return get_pipeline().calibrate()


@app.post("/api/scan")
async def api_scan(payload: ScanPayload) -> dict[str, Any]:
    """Mode foto: kirim gambar base64, terima kondisi + diskon."""
    raw = payload.image.split(",", 1)[-1]
    try:
        data = base64.b64decode(raw)
    except Exception:
        return JSONResponse({"ok": False, "error": "base64 tidak valid"}, status_code=400)
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"ok": False, "error": "gambar tidak dapat dibaca"}, status_code=400)
    return get_pipeline().analyse_image(frame, method=payload.source)


# ----------------------------------------------------------------------
# WebSocket: frame + telemetri
# ----------------------------------------------------------------------
@app.websocket("/ws")
async def ws_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    pipe = get_pipeline()
    try:
        while True:
            frame, state = await asyncio.to_thread(pipe.read)
            image_b64 = None
            frame_size = None
            if frame is not None:
                frame_size = [int(frame.shape[1]), int(frame.shape[0])]
                jpeg = await asyncio.to_thread(encode_jpeg, frame)
                if jpeg:
                    image_b64 = base64.b64encode(jpeg).decode("ascii")
            await websocket.send_json(
                {"type": "frame", "image": image_b64, "frame": frame_size, "state": state}
            )
            await asyncio.sleep(1.0 / 12.0)
    except WebSocketDisconnect:
        return
    except Exception as exc:  # pragma: no cover
        try:
            await websocket.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass


@app.on_event("shutdown")
async def shutdown() -> None:
    if _pipeline is not None:
        _pipeline.reader.stop()
