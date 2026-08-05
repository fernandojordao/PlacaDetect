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


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _feather_mask(
    roi_h: int, roi_w: int, inner: tuple[int, int, int, int], feather_px: int
) -> np.ndarray:
    """Máscara com a região da placa (`inner`) em opacidade total, esmaecendo suavemente até 0
    nas bordas da ROI.

    A "semente" (região em 1.0 antes do blur) é a caixa original *dilatada* por
    `feather_px` — não a caixa original sozinha. Isso garante que, depois do blur,
    o platô de opacidade total ainda cubra toda a placa (a suavização "come" a
    dilatação extra, não a placa em si), evitando tanto um degrau na borda da placa
    quanto um degrau na borda da ROI.
    """
    ix1, iy1, ix2, iy2 = inner
    feather_px = max(3, feather_px)

    seed = np.zeros((roi_h, roi_w), dtype=np.float32)
    sx1 = max(0, ix1 - feather_px)
    sy1 = max(0, iy1 - feather_px)
    sx2 = min(roi_w, ix2 + feather_px)
    sy2 = min(roi_h, iy2 + feather_px)
    seed[sy1:sy2, sx1:sx2] = 1.0

    ksize = _odd(feather_px * 2 + 1)
    mask = cv2.GaussianBlur(seed, (ksize, ksize), 0, borderType=cv2.BORDER_CONSTANT)

    # Reforço de segurança: a placa em si nunca deve ficar abaixo de opacidade total.
    mask[iy1:iy2, ix1:ix2] = np.maximum(mask[iy1:iy2, ix1:ix2], 0.98)
    return np.clip(mask, 0.0, 1.0)


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
        pad_x = max(2, int(box_w * padding_ratio))
        pad_y = max(2, int(box_h * padding_ratio))

        x1 = max(0, det.x1 - pad_x)
        y1 = max(0, det.y1 - pad_y)
        x2 = min(w, det.x2 + pad_x)
        y2 = min(h, det.y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            continue

        roi = out[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        if style == "pixelate":
            factor = 8
            small_w = max(1, roi_w // factor)
            small_h = max(1, roi_h // factor)
            small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            treated = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        elif style == "black":
            treated = np.full_like(roi, 20)
        else:  # "blur" (padrão) — uma única passada, proporcional à altura da placa.
            # Testado visualmente: abaixo de ~1.3x a altura da placa os caracteres
            # continuam parcialmente legíveis (inaceitável); acima de ~1.8x a placa
            # vira uma mancha praticamente sólida (perde a textura natural). Uma
            # única passada nessa faixa apaga o texto mas ainda deixa um gradiente
            # suave, em vez do efeito "chapado" de duas passadas somadas.
            k = _odd(min(71, max(15, box_h * 1.4)))
            treated = cv2.GaussianBlur(roi, (k, k), 0)

        # A região correspondente à caixa original (sem o padding) precisa ficar
        # sempre 100% coberta; o esmaecimento acontece só na margem de padding ao
        # redor — que agora é pequena de propósito — para que a transição se funda
        # com o resto da foto em vez de "colar" um retângulo artificial sobre a
        # placa ou criar um halo maior que ela.
        inner = (det.x1 - x1, det.y1 - y1, det.x2 - x1, det.y2 - y1)
        feather_px = max(2, min(pad_x, pad_y) // 2)
        mask = _feather_mask(roi_h, roi_w, inner, feather_px)[..., None]
        blended = roi.astype(np.float32) * (1 - mask) + treated.astype(np.float32) * mask
        out[y1:y2, x1:x2] = blended.astype(np.uint8)

    return out


def make_thumbnail(image_bgr: np.ndarray, path: Path, max_size: int = config.THUMBNAIL_MAX_SIZE) -> None:
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        image_bgr = cv2.resize(
            image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
        )
    save_image_bgr(image_bgr, path, quality=85)
