"""PlacaDetect — API FastAPI para detecção e redação automática de placas em fotos."""

from __future__ import annotations

import io
import logging
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, database as db, ingest, processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("placadetect")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="PlacaDetect", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurações de sessão (ajustáveis pela UI, em memória por enquanto).
_settings = {
    "model_name": config.DEFAULT_MODEL_NAME,
    "conf_thresh": config.DEFAULT_CONF_THRESH,
    "redaction_style": config.DEFAULT_REDACTION_STYLE,
}


# ---------------------------------------------------------------- schemas --

class ScanFolderRequest(BaseModel):
    folder: str
    recursive: bool = True


class ProcessRequest(BaseModel):
    scope: str = "pending"  # "pending" | "errors" | "all_ids"
    photo_ids: Optional[list[int]] = None


class SettingsRequest(BaseModel):
    model_name: Optional[str] = None
    conf_thresh: Optional[float] = None
    redaction_style: Optional[str] = None


# ------------------------------------------------------------------- API ---

@app.get("/api/stats")
def get_stats():
    return db.stats()


@app.get("/api/settings")
def get_settings():
    return _settings


@app.post("/api/settings")
def update_settings(req: SettingsRequest):
    if req.model_name is not None:
        _settings["model_name"] = req.model_name
    if req.conf_thresh is not None:
        _settings["conf_thresh"] = req.conf_thresh
    if req.redaction_style is not None:
        if req.redaction_style not in ("blur", "pixelate", "black"):
            raise HTTPException(400, "redaction_style deve ser blur, pixelate ou black")
        _settings["redaction_style"] = req.redaction_style
    return _settings


@app.post("/api/scan-folder")
def scan_folder(req: ScanFolderRequest):
    try:
        result = ingest.scan_folder(req.folder, req.recursive)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return result


@app.post("/api/upload")
async def upload_photos(files: list[UploadFile]):
    added = []
    errors = []
    for f in files:
        try:
            content = await f.read()
            photo_id = ingest.save_upload(f.filename or "foto.jpg", content)
            if photo_id is not None:
                added.append(photo_id)
        except ValueError as exc:
            errors.append({"filename": f.filename, "error": str(exc)})
    return {"added": len(added), "photo_ids": added, "errors": errors}


@app.get("/api/photos")
def list_photos(status: Optional[str] = None, page: int = 1, page_size: int = 60):
    photos, total = db.list_photos(status=status, page=page, page_size=page_size)
    for p in photos:
        _attach_urls(p)
    return {"photos": photos, "total": total, "page": page, "page_size": page_size}


@app.get("/api/photos/{photo_id}")
def get_photo(photo_id: int):
    photo = db.get_photo(photo_id)
    if not photo:
        raise HTTPException(404, "Foto não encontrada")
    _attach_urls(photo)
    return photo


@app.delete("/api/photos/{photo_id}")
def delete_photo(photo_id: int):
    photo = db.delete_photo(photo_id)
    if not photo:
        raise HTTPException(404, "Foto não encontrada")
    return {"deleted": photo_id}


@app.post("/api/photos/clear")
def clear_all_photos():
    count = db.clear_all()
    return {"deleted": count}


@app.post("/api/process")
def start_process(req: ProcessRequest):
    if processor.get_batch_state()["active"]:
        raise HTTPException(409, "Já existe um processamento em andamento")

    if req.scope == "pending":
        ids = db.list_pending_ids(("pending",))
    elif req.scope == "errors":
        ids = db.list_pending_ids(("error",))
    elif req.scope == "pending_and_errors":
        ids = db.list_pending_ids(("pending", "error"))
    elif req.scope == "ids":
        ids = req.photo_ids or []
        db.reset_to_pending(ids)
    else:
        raise HTTPException(400, "scope inválido")

    if not ids:
        return {"started": False, "reason": "Nenhuma foto pendente para processar", "total": 0}

    started = processor.start_batch_async(ids, dict(_settings))
    return {"started": started, "total": len(ids)}


@app.post("/api/process/cancel")
def cancel_process():
    processor.request_cancel()
    return {"ok": True}


@app.get("/api/process/status")
def process_status():
    state = processor.get_batch_state()
    return {**state, "counts": db.stats()}


@app.get("/api/download-zip")
def download_zip(status: str = "success"):
    photos, _ = db.list_photos(status=status, page=1, page_size=100000)
    photos = [p for p in photos if p.get("output_path")]
    if not photos:
        raise HTTPException(404, "Nenhuma foto processada encontrada para esse filtro")

    buffer = io.BytesIO()
    used_names: set[str] = set()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in photos:
            out_path = Path(p["output_path"])
            if not out_path.exists():
                continue
            name = p["filename"]
            if not name.lower().endswith((".jpg", ".jpeg")):
                name = Path(name).stem + ".jpg"
            if name in used_names:
                name = f"{p['id']}_{name}"
            used_names.add(name)
            zf.write(out_path, arcname=name)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="placadetect_{status}.zip"'},
    )


@app.get("/api/photos/{photo_id}/download")
def download_single(photo_id: int):
    photo = db.get_photo(photo_id)
    if not photo or not photo.get("output_path"):
        raise HTTPException(404, "Foto processada não encontrada")
    path = Path(photo["output_path"])
    if not path.exists():
        raise HTTPException(404, "Arquivo não encontrado em disco")
    return FileResponse(path, filename=photo["filename"])


def _attach_urls(photo: dict) -> None:
    photo["thumb_input_url"] = (
        f"/api/photos/{photo['id']}/thumb/input" if photo.get("thumb_input_path") else None
    )
    photo["thumb_output_url"] = (
        f"/api/photos/{photo['id']}/thumb/output" if photo.get("thumb_output_path") else None
    )
    photo["output_url"] = f"/api/photos/{photo['id']}/file/output" if photo.get("output_path") else None


@app.get("/api/photos/{photo_id}/thumb/{kind}")
def get_thumb(photo_id: int, kind: str):
    photo = db.get_photo(photo_id)
    if not photo:
        raise HTTPException(404, "Foto não encontrada")
    key = "thumb_input_path" if kind == "input" else "thumb_output_path"
    path = photo.get(key)
    if not path or not Path(path).exists():
        raise HTTPException(404, "Thumbnail não encontrada")
    return FileResponse(path)


@app.get("/api/photos/{photo_id}/file/{kind}")
def get_file(photo_id: int, kind: str):
    photo = db.get_photo(photo_id)
    if not photo:
        raise HTTPException(404, "Foto não encontrada")
    key = "output_path" if kind == "output" else "input_path"
    path = photo.get(key)
    if not path or not Path(path).exists():
        raise HTTPException(404, "Arquivo não encontrado")
    return FileResponse(path)


# ------------------------------------------------------------- frontend ---

if config.FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")
