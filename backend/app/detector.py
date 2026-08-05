"""Wrapper em torno do detector de placas (open-image-models / YOLOv9)."""

from __future__ import annotations

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


def detect_plates(
    image_bgr: np.ndarray,
    model_name: str | None = None,
    conf_thresh: float | None = None,
) -> list[PlateDetection]:
    model_name = model_name or config.DEFAULT_MODEL_NAME
    conf_thresh = conf_thresh if conf_thresh is not None else config.DEFAULT_CONF_THRESH
    detector = _get_detector(model_name, conf_thresh)
    results = detector.predict(image_bgr)

    detections: list[PlateDetection] = []
    for result in results:
        bbox = result.bounding_box
        detections.append(
            PlateDetection(
                x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=bbox.y2, confidence=float(result.confidence)
            )
        )
    return detections
