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


def test_pad_body_polygon_never_shrinks_and_preserves_header_edge():
    img, quad, (x0, y0) = _synthetic_plate(width=200, height=160, header_height=30)
    body = pg.find_header_cut(img, quad)
    padded = pg.pad_body_polygon(body, 0.08)

    # a borda "de baixo" (fim real da placa) deve se expandir pra fora
    assert padded[:, 1].max() > body[:, 1].max()
    # a borda esquerda/direita deve se expandir pra fora
    assert padded[:, 0].min() < body[:, 0].min()
    assert padded[:, 0].max() > body[:, 0].max()
    # a borda do corte (perto do cabeçalho) NÃO deve avançar sobre o
    # cabeçalho - o topo do polígono expandido não pode ficar acima do corte
    # original (senão estaria comendo de volta a faixa que devia ficar visível)
    assert padded[:, 1].min() >= body[:, 1].min() - 1e-3
