"""Configurações centrais do PlacaDetect."""

from __future__ import annotations

import os
from pathlib import Path

# Raiz do projeto (PlacaDetect/)
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("PLACADETECT_DATA_DIR", BASE_DIR / "data")).resolve()
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
THUMBS_DIR = DATA_DIR / "thumbs"
DB_PATH = DATA_DIR / "placadetect.db"

FRONTEND_DIR = BASE_DIR / "frontend"

for _dir in (INPUT_DIR, OUTPUT_DIR, THUMBS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Modelo de detecção de placas (open-image-models / YOLOv9 "end2end").
# Modelos disponíveis (do mais leve/rápido ao mais preciso):
#   yolo-v9-t-256-license-plate-end2end
#   yolo-v9-t-384-license-plate-end2end
#   yolo-v9-t-416-license-plate-end2end
#   yolo-v9-t-512-license-plate-end2end
#   yolo-v9-t-640-license-plate-end2end
#   yolo-v9-s-608-license-plate-end2end  (mais preciso)
DEFAULT_MODEL_NAME = os.environ.get("PLACADETECT_MODEL", "yolo-v9-t-640-license-plate-end2end")
DEFAULT_CONF_THRESH = float(os.environ.get("PLACADETECT_CONF_THRESH", "0.3"))

# Detecções com proporção largura/altura acima desse teto são descartadas (mesmo
# acima do limiar de confiança) — nenhuma placa real, nem em ângulo forte, fica
# tão alongada quanto isso; esse extremo é sinal de faixa/friso comprido na moto.
PLATE_ASPECT_RATIO_RANGE = (1.0, 6.0)

# Placas de moto (Mercosul) são quase quadradas (~1.2–1.3) e, fotografadas em
# ângulo, a projeção pode ficar ainda mais próxima de um quadrado perfeito — não
# dá pra usar "é quase quadrado" sozinho para rejeitar, senão placas reais em
# ângulo são descartadas junto com adesivos/refletores quadrados/redondos (esse
# foi o bug: uma moto fotografada de lado ficava com "sucesso" mas sem nenhuma
# placa redigida, porque a única detecção real foi jogada fora aqui). Abaixo
# dessa proporção, só aceitamos a detecção se a confiança for razoável — um
# adesivo/refletor tende a ter confiança bem mais baixa que uma placa de verdade.
PLATE_SQUARE_RATIO_GUARD = 1.35
PLATE_SQUARE_MIN_CONF = 0.45

# Estilo de redação aplicado sobre a placa detectada: "blur" | "pixelate" | "black"
DEFAULT_REDACTION_STYLE = os.environ.get("PLACADETECT_REDACTION_STYLE", "blur")

# Expande a caixa detectada em X% para garantir cobertura total da placa
# (o detector é axis-aligned; motos inclinadas podem ter a placa levemente
# fora da caixa se não houver essa margem) e para sobrar uma margem mínima
# onde a borda do blur possa suavizar. Mantido pequeno de propósito: a área
# tratada deve acompanhar o tamanho real da placa, não um halo grande ao
# redor dela — senão o resultado chama mais atenção do que a própria placa.
BOX_PADDING_RATIO = float(os.environ.get("PLACADETECT_BOX_PADDING", "0.12"))

THUMBNAIL_MAX_SIZE = 480

JPEG_QUALITY = 92
