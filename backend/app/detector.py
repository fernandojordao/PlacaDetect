"""Wrapper em torno do detector de placas (open-image-models / YOLOv9).

O modelo redimensiona a imagem inteira para sua resolução de entrada (ex.: 640x640)
antes de detectar. Em fotos grandes com várias motos no quadro (comum em fotos de
estacionamento/pista), as placas mais distantes podem encolher a poucos pixels
nessa redução e passar despercebidas. Para compensar, além de rodar a detecção na
imagem inteira, também rodamos em recortes (tiles) sobrepostos do tamanho nativo do
modelo quando a imagem é bem maior que essa resolução — isso preserva a escala
original das placas menores. As detecções de todas as passadas são combinadas com
NMS para remover duplicatas.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass

import numpy as np
from open_image_models import create_detector

from . import config

_lock = threading.Lock()
_detector = None
_detector_key: tuple[str, float] | None = None


@dataclass
class PlateDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    def to_dict(self) -> dict:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": round(self.confidence, 4),
        }

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


def _get_detector(model_name: str, conf_thresh: float):
    global _detector, _detector_key
    key = (model_name, conf_thresh)
    with _lock:
        if _detector is None or _detector_key != key:
            _detector = create_detector(model_name, conf_thresh=conf_thresh)
            _detector_key = key
        return _detector


def warmup(model_name: str | None = None, conf_thresh: float | None = None) -> None:
    """Carrega (e baixa, se necessário) o modelo antes do primeiro uso real."""
    model_name = model_name or config.DEFAULT_MODEL_NAME
    conf_thresh = conf_thresh if conf_thresh is not None else config.DEFAULT_CONF_THRESH
    detector = _get_detector(model_name, conf_thresh)
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    detector.predict(dummy)


def _model_input_size(model_name: str) -> int:
    match = re.search(r"-(\d{3,4})-", model_name)
    return int(match.group(1)) if match else 640


def _run_pass(detector, image: np.ndarray, offset_x: int, offset_y: int) -> list[PlateDetection]:
    results = detector.predict(image)
    detections = []
    for result in results:
        bbox = result.bounding_box
        detections.append(
            PlateDetection(
                x1=bbox.x1 + offset_x,
                y1=bbox.y1 + offset_y,
                x2=bbox.x2 + offset_x,
                y2=bbox.y2 + offset_y,
                confidence=float(result.confidence),
            )
        )
    return detections


def _tile_boxes(width: int, height: int, tile: int, overlap: float = 0.25) -> list[tuple[int, int, int, int]]:
    if width <= tile and height <= tile:
        return []

    def _positions(size: int) -> list[int]:
        step = max(1, int(tile * (1 - overlap)))
        pos = list(range(0, max(1, size - tile) + 1, step))
        if not pos or pos[-1] != max(0, size - tile):
            pos.append(max(0, size - tile))
        return pos

    boxes = []
    for y in _positions(height):
        for x in _positions(width):
            boxes.append((x, y, min(x + tile, width), min(y + tile, height)))
    return boxes


def _iou(a: PlateDetection, b: PlateDetection) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _merge_detections(detections: list[PlateDetection], iou_thresh: float = 0.35) -> list[PlateDetection]:
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[PlateDetection] = []
    for det in ordered:
        if all(_iou(det, k) < iou_thresh for k in kept):
            kept.append(det)
    return kept


def detect_plates(
    image_bgr: np.ndarray,
    model_name: str | None = None,
    conf_thresh: float | None = None,
    tiled: bool = True,
) -> list[PlateDetection]:
    model_name = model_name or config.DEFAULT_MODEL_NAME
    conf_thresh = conf_thresh if conf_thresh is not None else config.DEFAULT_CONF_THRESH
    detector = _get_detector(model_name, conf_thresh)

    height, width = image_bgr.shape[:2]
    all_detections = _run_pass(detector, image_bgr, 0, 0)

    if tiled:
        tile_size = _model_input_size(model_name)
        if max(width, height) > tile_size * 1.35:
            for (tx1, ty1, tx2, ty2) in _tile_boxes(width, height, tile_size):
                crop = image_bgr[ty1:ty2, tx1:tx2]
                if crop.shape[0] < 48 or crop.shape[1] < 48:
                    continue
                all_detections.extend(_run_pass(detector, crop, tx1, ty1))

    return _merge_detections(all_detections)
