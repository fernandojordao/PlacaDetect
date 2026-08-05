"""Testes da geometria fina de placa (backend/app/plate_geometry.py).

Cobre especificamente os dois bugs reais encontrados testando contra fotos de
moto de verdade:
1. O corte de cabeçalho cortando o eixo errado (largura em vez de altura).
2. A mesma coisa reaparecendo só em placas quase quadradas, por causa da
   ambiguidade de qual lado o minAreaRect considera "comprido".
"""

import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app import plate_geometry as pg  # noqa: E402


def _synthetic_plate(width, height, header_height):
    """Placa sintética: faixa azul (estilo Mercosul) no topo, corpo branco
    abaixo. Retorna (imagem, quad axis-aligned)."""
    img = np.full((height + 120, width + 120, 3), 40, dtype=np.uint8)
    x0, y0 = 60, 60
    cv2.rectangle(img, (x0, y0), (x0 + width, y0 + header_height), (200, 140, 30), -1)
    cv2.rectangle(img, (x0, y0 + header_height), (x0 + width, y0 + height), (230, 230, 230), -1)
    quad = np.array(
        [[x0, y0], [x0 + width, y0], [x0 + width, y0 + height], [x0, y0 + height]],
        dtype=np.float32,
    )
    return img, quad, (x0, y0)


def test_header_cut_wide_plate_preserves_full_width_and_cuts_height():
    img, quad, (x0, y0) = _synthetic_plate(width=200, height=160, header_height=30)
    body = pg.find_header_cut(img, quad)
    assert body is not None

    xs, ys = body[:, 0], body[:, 1]
    # a largura toda deve continuar coberta (o corte é na altura, não na largura)
    assert xs.min() == x0
    assert xs.max() == x0 + 200
    # o topo do corpo deve ficar por volta do fim do cabeçalho (30px), nunca
    # mais alto que isso (senão sobraria parte do cabeçalho borrada por engano
    # não é o problema de segurança aqui, mas indicaria o corte errado)
    assert y0 + 20 < ys.min() < y0 + 32
    assert ys.max() == y0 + 160


def test_header_cut_near_square_plate_still_cuts_height_not_width():
    """Pin de regressão: com w=180,h=185 (quase quadrado), minAreaRect pode
    devolver o lado comprido/curto invertido por causa de arredondamento, e
    o corte antigo (baseado só em geometria) acabava cortando a largura,
    deixando o cabeçalho inteiro dentro da área borrada."""
    img, quad, (x0, y0) = _synthetic_plate(width=180, height=185, header_height=35)
    body = pg.find_header_cut(img, quad)
    assert body is not None

    xs, ys = body[:, 0], body[:, 1]
    assert xs.min() == x0
    assert xs.max() == x0 + 180
    assert y0 + 25 < ys.min() < y0 + 37
    assert ys.max() == y0 + 185


def test_header_cut_returns_none_without_blue_band():
    """Placa sem faixa azul (padrão antigo, ou placa suja) deve cair no
    fallback seguro (None => borra a placa inteira), nunca inventar um corte."""
    img = np.full((260, 320, 3), 230, dtype=np.uint8)  # tudo branco/cinza, sem azul
    quad = np.array([[60, 60], [260, 60], [260, 220], [60, 220]], dtype=np.float32)
    assert pg.find_header_cut(img, quad) is None


def _rotated_synthetic_plate(width, height, header_height, angle_deg, canvas=420):
    """Mesma placa sintética, mas desenhada já rotacionada no canvas (rotaciona
    a arte inteira, não só os pontos) — testa o pipeline completo de detecção
    de ângulo via cor, não só a matemática de reconstrução do quad."""
    plate = np.zeros((height, width, 3), dtype=np.uint8)
    plate[:header_height] = (200, 140, 30)
    plate[header_height:] = (230, 230, 230)

    img = np.full((canvas, canvas, 3), 40, dtype=np.uint8)
    cx, cy = canvas // 2, canvas // 2
    corners_local = np.array(
        [[-width / 2, -height / 2], [width / 2, -height / 2], [width / 2, height / 2], [-width / 2, height / 2]],
        dtype=np.float32,
    )
    theta = np.radians(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    dst_corners = (corners_local @ rot.T) + np.array([cx, cy], dtype=np.float32)
    src_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    warped = cv2.warpPerspective(plate, M, (canvas, canvas))
    mask = cv2.warpPerspective(np.full((height, width), 255, dtype=np.uint8), M, (canvas, canvas))
    img[mask > 0] = warped[mask > 0]

    axis_box = cv2.boundingRect(dst_corners.astype(np.int32))
    return img, dst_corners, axis_box


def test_find_plate_quad_recovers_tilt_from_header_band():
    """A caixa axis-aligned do detector (bounding box da placa rotacionada)
    é sempre maior/imprecisa; find_plate_quad deve reconstruir um
    quadrilátero que acompanha a inclinação real, não a caixa esticada."""
    width, height, header_h, angle = 200, 160, 32, 18.0
    img, true_corners, (bx, by, bw, bh) = _rotated_synthetic_plate(width, height, header_h, angle)

    quad = pg.find_plate_quad(img, bx, by, bx + bw, by + bh)
    assert quad is not None

    # a área do quad reconstruído deve ficar perto da área real da placa
    # (bem menor que a da caixa axis-aligned, que é inflada pela rotação)
    true_area = width * height
    axis_area = bw * bh
    quad_area = cv2.contourArea(quad.astype(np.float32))
    assert axis_area > true_area * 1.05  # confirma que a caixa really infla a área
    assert abs(quad_area - true_area) / true_area < 0.15

    # cada canto reconstruído deve estar perto de algum canto verdadeiro
    for corner in quad:
        dists = np.linalg.norm(true_corners - corner, axis=1)
        assert dists.min() < 0.08 * max(width, height)
