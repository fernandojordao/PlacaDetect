"""Utilitários de imagem: carregamento seguro (EXIF), redação de placas e thumbnails."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import config, plate_geometry as pgeo
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


def _feather_mask_poly(
    roi_h: int, roi_w: int, core_poly: np.ndarray, outer_poly: np.ndarray, feather_px: int
) -> np.ndarray:
    """Máscara com a região `core_poly` em opacidade total, esmaecendo
    suavemente até 0 fora de `outer_poly`.

    Versão poligonal de `_feather_mask` (mesma ideia, generalizada pra
    acompanhar um retângulo rotacionado/recortado em vez de só axis-aligned):
    a "semente" é `outer_poly` (a área já com a margem de padding), borrada
    pra suavizar a borda; `core_poly` (a placa/corpo real, sem padding) é
    reforçada pra nunca ficar abaixo de opacidade quase total, garantindo que
    o platô cheio sempre cubra pelo menos a área que realmente precisa ficar
    ilegível.
    """
    feather_px = max(3, feather_px)

    seed = np.zeros((roi_h, roi_w), dtype=np.float32)
    cv2.fillPoly(seed, [np.round(outer_poly).astype(np.int32)], 1.0)

    ksize = _odd(feather_px * 2 + 1)
    mask = cv2.GaussianBlur(seed, (ksize, ksize), 0, borderType=cv2.BORDER_CONSTANT)

    core_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillPoly(core_mask, [np.round(core_poly).astype(np.int32)], 1)
    mask = np.where(core_mask > 0, np.maximum(mask, 0.98), mask)
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

        # Quad "de trabalho": tenta achar o contorno real (rotacionado) da
        # placa; se não achar com confiança, cai pra caixa axis-aligned do
        # detector — sempre seguro, só perde o acompanhamento da inclinação.
        base_quad = pgeo.find_plate_quad(image_bgr, det.x1, det.y1, det.x2, det.y2)
        if base_quad is None:
            base_quad = np.array(
                [[det.x1, det.y1], [det.x2, det.y1], [det.x2, det.y2], [det.x1, det.y2]],
                dtype=np.float32,
            )

        # Se a faixa de cabeçalho (azul, padrão Mercosul) for identificada com
        # confiança, borra só o "corpo" (letras/números) e deixa o cabeçalho
        # visível. Sem confiança nenhuma, borra a placa inteira — mais seguro
        # do que arriscar um corte errado deixando caracteres de fora.
        core_poly = pgeo.find_header_cut(image_bgr, base_quad)
        if core_poly is not None:
            outer_poly = pgeo.pad_body_polygon(core_poly, padding_ratio)
        else:
            core_poly = base_quad
            outer_poly = pgeo.pad_quad(base_quad, padding_ratio)

        rx1, ry1, rw, rh = cv2.boundingRect(np.round(outer_poly).astype(np.int32))
        x1, y1 = max(0, rx1), max(0, ry1)
        x2, y2 = min(w, rx1 + rw), min(h, ry1 + rh)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = out[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]
        offset = np.array([x1, y1], dtype=np.float32)
        core_local = core_poly - offset
        outer_local = outer_poly - offset

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
            # suave, em vez do efeito "chapado" de duas passadas somadas. O kernel
            # usa a altura da caixa detectada inteira (não do corpo, que pode ser
            # menor por causa do corte do cabeçalho) — a escala do borrão segue o
            # tamanho real da placa, não da fração que sobrou pra borrar.
            k = _odd(min(71, max(15, box_h * 1.4)))
            treated = cv2.GaussianBlur(roi, (k, k), 0)

        # O corpo real da placa (sem padding) precisa ficar sempre 100%
        # coberto; o esmaecimento acontece só na margem de padding ao redor
        # — que agora é pequena de propósito — para que a transição se funda
        # com o resto da foto em vez de "colar" um retângulo artificial sobre
        # a placa ou criar um halo maior que ela.
        min_edge = min(box_w, box_h)
        feather_px = max(2, int(min_edge * padding_ratio * 0.5))
        mask = _feather_mask_poly(roi_h, roi_w, core_local, outer_local, feather_px)[..., None]
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
