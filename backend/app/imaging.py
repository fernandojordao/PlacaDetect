"""Utilitários de imagem: carregamento seguro (EXIF), redação de placas e thumbnails."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import config
from .detector import PlateDetection


def load_image_bgr(path: Path) -> np.ndarray:
    """Carrega a imagem já corrigindo a orientação EXIF (comum em fotos de celular)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        rgb = np.array(im)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_image_bgr(image_bgr: np.ndarray, path: Path, quality: int = config.JPEG_QUALITY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    save_kwargs = {}
    if path.suffix.lower() in (".jpg", ".jpeg"):
        save_kwargs = {"quality": quality, "optimize": True}
    im.save(path, **save_kwargs)


def redact_plates(
    image_bgr: np.ndarray,
    detections: list[PlateDetection],
    style: str = config.DEFAULT_REDACTION_STYLE,
    padding_ratio: float = config.BOX_PADDING_RATIO,
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]

    for det in detections:
        box_w = det.x2 - det.x1
        box_h = det.y2 - det.y1
        pad_x = int(box_w * padding_ratio)
        pad_y = int(box_h * padding_ratio)

        x1 = max(0, det.x1 - pad_x)
        y1 = max(0, det.y1 - pad_y)
        x2 = min(w, det.x2 + pad_x)
        y2 = min(h, det.y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            continue

        roi = out[y1:y2, x1:x2]

        if style == "pixelate":
            factor = 10
            small_w = max(1, roi.shape[1] // factor)
            small_h = max(1, roi.shape[0] // factor)
            small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            roi = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
        elif style == "black":
            roi = np.full_like(roi, 20)
        else:  # "blur" (padrão)
            k = int(min(roi.shape[0], roi.shape[1]) * 0.9)
            k = max(31, k)
            if k % 2 == 0:
                k += 1
            roi = cv2.GaussianBlur(roi, (k, k), 0)

        out[y1:y2, x1:x2] = roi

    return out


def make_thumbnail(image_bgr: np.ndarray, path: Path, max_size: int = config.THUMBNAIL_MAX_SIZE) -> None:
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        image_bgr = cv2.resize(
            image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
        )
    save_image_bgr(image_bgr, path, quality=85)
