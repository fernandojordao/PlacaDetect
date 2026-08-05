"""Utilitários de imagem: carregamento seguro (EXIF), redação de placas e thumbnails."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rawpy
from PIL import Image, ImageOps

from . import config, plate_geometry as pgeo
from .detector import PlateDetection


def load_image_bgr(path: Path) -> np.ndarray:
    """Carrega a imagem já corrigindo a orientação EXIF (comum em fotos de celular)."""
    if path.suffix.lower() in config.RAW_EXTENSIONS:
        return _load_raw_bgr(path)
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        rgb = np.array(im)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _load_raw_bgr(path: Path) -> np.ndarray:
    """Decodifica um RAW (.CR3) na resolução nativa do sensor via LibRaw.

    Faz a demosaicagem completa (`postprocess`) em vez de só extrair o
    preview/thumbnail embutido no arquivo — o preview costuma vir numa
    resolução bem menor que o sensor, e o usuário quer especificamente a
    qualidade total do RAW (é o motivo de fotografar em RAW em primeiro
    lugar). O balanço de branco "as-shot" da câmera é usado por padrão
    (`use_camera_wb`), pra não alterar a aparência da foto — o objetivo aqui
    é só decodificar e depois redigir a placa, não editar a foto.
    """
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
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


def _feather_mask_poly(roi_h: int, roi_w: int, poly: np.ndarray, feather_px: int) -> np.ndarray:
    """Máscara em opacidade total dentro de `poly`, esmaecendo suavemente só
    para DENTRO perto da borda — nunca se estende além do polígono.

    Um blur gaussiano comum de uma máscara binária espalha pros dois lados da
    borda (a região tratada acabaria maior que `poly`, que é exatamente o que
    não pode acontecer: o desfoque tem que ficar contido na área real da
    placa, sem sobrar em cima do que está ao redor dela). Truncar o blur pelo
    valor original da máscara (`np.minimum`) garante isso: fora de `poly` o
    valor da semente já era 0 e continua 0; dentro, o blur só pode reduzir a
    opacidade perto da borda, nunca "vazar" pra fora dela.
    """
    feather_px = max(1, feather_px)

    seed = np.zeros((roi_h, roi_w), dtype=np.float32)
    cv2.fillPoly(seed, [np.round(poly).astype(np.int32)], 1.0)

    ksize = _odd(feather_px * 2 + 1)
    blurred = cv2.GaussianBlur(seed, (ksize, ksize), 0, borderType=cv2.BORDER_CONSTANT)
    return np.minimum(seed, blurred)


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
        if core_poly is None:
            core_poly = base_quad

        # A região tratada é EXATAMENTE `core_poly` — sem margem pra fora
        # dele. Nada de padding aqui: o usuário quer o desfoque contido só na
        # área real da placa, acompanhando seu contorno/inclinação, nunca
        # "vazando" pra cima do que está ao redor (guidão, pneu, parede).
        rx1, ry1, rw, rh = cv2.boundingRect(np.round(core_poly).astype(np.int32))
        x1, y1 = max(0, rx1), max(0, ry1)
        x2, y2 = min(w, rx1 + rw), min(h, ry1 + rh)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = out[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]
        offset = np.array([x1, y1], dtype=np.float32)
        core_local = core_poly - offset

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

        # O esmaecimento acontece só PRA DENTRO, perto da borda de `core_poly`
        # — nunca estica a área tratada além do contorno real da placa (ver
        # `_feather_mask_poly`). `padding_ratio` aqui não é mais uma margem
        # geométrica: é só a espessura relativa dessa faixa de transição
        # (fração do menor lado da placa), pra borda não ficar um corte seco.
        min_edge = min(box_w, box_h)
        feather_px = max(2, int(min_edge * padding_ratio))
        mask = _feather_mask_poly(roi_h, roi_w, core_local, feather_px)[..., None]
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
