"""Entrada de fotos no sistema: importação de pasta local e upload via navegador."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from . import config, database as db


def scan_folder(folder: str, recursive: bool = True) -> dict[str, Any]:
    root = Path(folder).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {folder}")
    if not root.is_dir():
        raise NotADirectoryError(f"Não é uma pasta: {folder}")

    pattern = "**/*" if recursive else "*"
    found = 0
    added = 0
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        if path.suffix.lower() not in config.IMAGE_EXTENSIONS:
            continue
        found += 1
        photo_id = db.insert_photo(path.name, str(path.resolve()), source="folder")
        if photo_id is not None:
            added += 1

    return {"found": found, "added": added, "skipped_existing": found - added}


def save_upload(filename: str, content: bytes) -> int | None:
    safe_name = Path(filename).name
    ext = Path(safe_name).suffix.lower()
    if ext not in config.IMAGE_EXTENSIONS:
        raise ValueError(f"Extensão não suportada: {ext}")

    unique_name = f"{uuid.uuid4().hex[:10]}_{safe_name}"
    dest = config.INPUT_DIR / unique_name
    dest.write_bytes(content)

    photo_id = db.insert_photo(safe_name, str(dest.resolve()), source="upload")
    if photo_id is None:
        dest.unlink(missing_ok=True)
    return photo_id
