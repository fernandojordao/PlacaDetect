"""Wrapper em torno do detector de placas (open-image-models / YOLOv9).

O modelo redimensiona a imagem inteira para sua resolução de entrada (ex.: 640x640)
antes de detectar. Em fotos grandes (comum em fotos de celular, várias motos no
quadro, placas distantes/em ângulo), isso tem duas consequências:

1. Uma placa pequena pode encolher a poucos pixels nessa redução e passar
   despercebida — por isso também rodamos a detecção em recortes (tiles)
   sobrepostos no tamanho nativo do modelo.
2. A caixa devolvida pela passada "imagem inteira" tende a ser geometricamente
   imprecisa quando o objeto era minúsculo na versão redimensionada (o erro de
   poucos pixels em 640px vira um erro de dezenas de pixels na foto original).
   As caixas vindas de um recorte (tile) são bem mais precisas, pois o recorte
   já está perto da resolução nativa do modelo. Por isso, na hora de combinar
   detecções, uma caixa de recorte sempre tem prioridade sobre uma caixa da
   passada de imagem inteira que se sobreponha a ela — evita "sucesso" com o
   desfoque caindo ao lado da placa em vez de em cima.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass

import numpy as np
from open_image_models import create_detector

from . import config

_lock = threading.Lock()
_detectors: dict[tuple[str, float], object] = {}


@dataclass
class PlateDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    source: str = "full"  # "full" (imagem inteira) ou "tile" (recorte nativo)

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
    key = (model_name, conf_thresh)
    with _lock:
        detector = _detectors.get(key)
        if detector is None:
            detector = create_detector(model_name, conf_thresh=conf_thresh)
            _detectors[key] = detector
        return detector


def warmup(model_name: str | None = None, conf_thresh: float | None = None) -> None:
    """Carrega (e baixa, se necessário) os modelos antes do primeiro uso real."""
    model_name = model_name or config.DEFAULT_MODEL_NAME
    conf_thresh = conf_thresh if conf_thresh is not None else config.DEFAULT_CONF_THRESH
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    _get_detector(model_name, conf_thresh).predict(dummy)
    _get_detector(model_name, _tile_conf_thresh(conf_thresh)).predict(dummy)


def _tile_conf_thresh(base_conf: float) -> float:
    """Recortes veem a placa em resolução mais próxima da nativa, então dá pra ser
    mais permissivo sem tanto risco de falso positivo — ajuda a recall em placas
    pequenas/em ângulo sem depender de afrouxar o limiar da imagem inteira (que aí
    sim tende a gerar mais falso positivo, por ver a foto toda em baixa resolução).
    O filtro de formato (`_plausible_plate_shape`) é a segunda linha de defesa
    contra falso positivo, então dá pra ser mais permissivo aqui do que pareceria
    seguro isoladamente."""
    return max(0.10, round(base_conf * 0.5, 3))


def _model_input_size(model_name: str) -> int:
    match = re.search(r"-(\d{3,4})-", model_name)
    return int(match.group(1)) if match else 640


def _run_pass(detector, image: np.ndarray, offset_x: int, offset_y: int, source: str) -> list[PlateDetection]:
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
                source=source,
            )
        )
    return detections


def _tile_boxes(width: int, height: int, tile: int, overlap: float = 0.35) -> list[tuple[int, int, int, int]]:
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


def _plausible_plate_shape(det: PlateDetection) -> bool:
    box_w = det.x2 - det.x1
    box_h = det.y2 - det.y1
    if box_w <= 0 or box_h <= 0:
        return False
    ratio = max(box_w, box_h) / min(box_w, box_h)
    lo, hi = config.PLATE_ASPECT_RATIO_RANGE
    if not (lo <= ratio <= hi):
        return False
    if ratio < config.PLATE_SQUARE_RATIO_GUARD:
        return det.confidence >= config.PLATE_SQUARE_MIN_CONF
    return True


def _overlap_ratio(a: PlateDetection, b: PlateDetection) -> float:
    """Fração da área do menor dos dois boxes coberta pela interseção.

    Diferente do IoU, isso não penaliza um casamento óbvio só porque os dois boxes
    têm tamanhos bem diferentes — que é exatamente o caso aqui: o box da passada de
    imagem inteira é impreciso (às vezes bem maior ou deslocado em relação à placa
    real), enquanto o box de um tile é justo. Dois boxes assim, apontando para a
    mesma placa, podem ter IoU baixo mesmo sendo o mesmo objeto — e com IoU eles
    passavam os dois como "detecções distintas", dobrando a área borrada em cima de
    uma única placa (e enganando o usuário sobre haver 2 placas).
    """
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    smaller = min(a.area, b.area)
    return inter / smaller if smaller > 0 else 0.0


def _merge_detections(detections: list[PlateDetection], overlap_thresh: float = 0.35) -> list[PlateDetection]:
    """Combina detecções priorizando fonte (tile > full) e depois confiança.

    Qualquer sobreposição relevante entre uma caixa de tile e uma de imagem inteira
    faz a de imagem inteira ser descartada — ela é a candidata menos precisa das
    duas, então nunca deve "vencer" nem ficar como duplicata ao lado da versão
    correta (ver `_overlap_ratio`).
    """
    plausible = [d for d in detections if _plausible_plate_shape(d)]
    tiles = sorted((d for d in plausible if d.source == "tile"), key=lambda d: d.confidence, reverse=True)
    fulls = sorted((d for d in plausible if d.source == "full"), key=lambda d: d.confidence, reverse=True)

    kept: list[PlateDetection] = []
    for det in tiles:
        if all(_overlap_ratio(det, k) < overlap_thresh for k in kept):
            kept.append(det)

    for det in fulls:
        if all(_overlap_ratio(det, k) < overlap_thresh for k in kept):
            kept.append(det)

    return kept



# Resolução "de referência" em que a estratégia de tiling (recorte no
# tamanho nativo do modelo) foi testada e validada contra fotos reais. Fotos
# MUITO maiores que isso (RAW de câmera/.CR3, que decodificam na resolução
# do sensor — várias vezes maior que uma foto de celular) recebem também uma
# segunda passada com tiles proporcionalmente maiores (ver `detect_plates`).
_TILING_REFERENCE_LONG_SIDE = 1600


def detect_plates(
    image_bgr: np.ndarray,
    model_name: str | None = None,
    conf_thresh: float | None = None,
    tiled: bool = True,
) -> list[PlateDetection]:
    model_name = model_name or config.DEFAULT_MODEL_NAME
    conf_thresh = conf_thresh if conf_thresh is not None else config.DEFAULT_CONF_THRESH

    height, width = image_bgr.shape[:2]
    full_detector = _get_detector(model_name, conf_thresh)
    all_detections = _run_pass(full_detector, image_bgr, 0, 0, source="full")

    if tiled:
        native_tile = _model_input_size(model_name)
        long_side = max(width, height)
        if long_side > native_tile * 1.35:
            tile_detector = _get_detector(model_name, _tile_conf_thresh(conf_thresh))
            tile_sizes = {native_tile}

            # Recorte no tamanho nativo tem uma desvantagem em fotos MUITO
            # maiores que a referência: cada tile passa a cobrir uma fatia
            # proporcionalmente bem menor da cena do que cobria numa foto de
            # celular — o oposto do problema original (placa pequena demais
            # na passada de imagem inteira), mas com o mesmo efeito líquido:
            # detecção que funciona numa resolução e falha em outra, mesmo
            # sendo exatamente a mesma cena/placa. Escalar o tile junto com a
            # resolução da imagem mantém a fração da cena por tile parecida
            # em qualquer resolução — o próprio `predict()` já redimensiona
            # o recorte pro tamanho de entrada do modelo por baixo dos panos,
            # então um tile maior aqui não muda o custo do modelo em si, só
            # quantos tiles cabem na imagem.
            if long_side > _TILING_REFERENCE_LONG_SIDE * 1.5:
                scale = long_side / _TILING_REFERENCE_LONG_SIDE
                tile_sizes.add(int(native_tile * scale))

            for tile_size in tile_sizes:
                for (tx1, ty1, tx2, ty2) in _tile_boxes(width, height, tile_size):
                    crop = image_bgr[ty1:ty2, tx1:tx2]
                    if crop.shape[0] < 48 or crop.shape[1] < 48:
                        continue
                    all_detections.extend(_run_pass(tile_detector, crop, tx1, ty1, source="tile"))

    return _merge_detections(all_detections)
