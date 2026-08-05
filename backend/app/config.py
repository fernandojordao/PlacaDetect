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

# Detecções com proporção largura/altura fora dessa faixa são descartadas (mesmo
# acima do limiar de confiança) — placas reais, mesmo fotografadas em ângulo, não
# ficam próximas de um quadrado nem extremamente alongadas. Isso filtra falsos
# positivos como adesivos/refletores redondos ou quadrados na moto.
PLATE_ASPECT_RATIO_RANGE = (1.15, 6.0)

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
