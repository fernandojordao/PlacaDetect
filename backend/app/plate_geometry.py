"""Geometria fina da placa dentro da caixa detectada pelo modelo.

O detector (YOLO) só devolve uma caixa axis-aligned. Duas coisas que o usuário
pediu não dá pra fazer só com isso:

1. O desfoque acompanhar o contorno/inclinação real da placa (uma moto
   inclinada ou fotografada em ângulo projeta a placa como um retângulo
   rotacionado, não axis-aligned; usar a caixa do detector direto sempre borra
   área de fundo a mais nos cantos).
2. Preservar a faixa azul "BRASIL" (padrão Mercosul) e borrar só a parte com
   letras/números.

As duas coisas aqui são heurísticas de visão computacional clássica (contorno
+ HSV), não vêm do modelo. Por isso todo o módulo é "best-effort com fallback
seguro": qualquer situação ambígua devolve None e quem chama cai de volta no
comportamento antigo (caixa axis-aligned inteira) — é preferível borrar uma
faixa saudável de fundo a mais do que arriscar, por uma detecção de contorno
errada, deixar caracteres da placa de fora do desfoque.
"""

from __future__ import annotations

import cv2
import numpy as np

# Faixa HSV do azul Mercosul, calibrada em cima de fotos reais (ver commit).
_BLUE_HUE_RANGE = (95, 125)
_BLUE_SAT_MIN = 60
_BLUE_VAL_MIN = 40

# Fração mínima de pixels azuis numa faixa fina, perto de uma das pontas do
# lado curto da placa, para considerar que ali tem cabeçalho Mercosul.
_HEADER_MIN_BLUE_FRAC = 0.12

# Nunca deixamos a faixa "preservada" passar de 45% da altura da placa — isso
# limita o dano de um falso positivo de azul (reflexo, adesivo azul etc.)
# tomando conta de mais da metade da placa.
_HEADER_MAX_FRAC = 0.45

# Margem de segurança: encolhe a faixa preservada um pouco em direção ao
# cabeçalho, então o corte fica sempre um pouco "dentro" do azul detectado —
# garante que nenhum pixel de letra fique perto o bastante da borda do
# cabeçalho para escapar do desfoque por causa de uma medição imprecisa.
_HEADER_SAFETY_MARGIN_FRAC = 0.04


def _order_box_points(pts: np.ndarray) -> np.ndarray:
    """cv2.boxPoints já devolve 4 pontos em ordem cíclica consistente (ver
    documentação do OpenCV) — só garantimos o tipo/formato aqui."""
    return pts.astype(np.float32)


def find_plate_quad(image_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray | None:
    """Tenta achar o quadrilátero real da placa (rotacionado) dentro/perto da
    caixa detectada. Devolve 4 pontos (float32, ordem cíclica) em coordenadas
    da imagem original, ou None se nada plausível for encontrado.
    """
    h, w = image_bgr.shape[:2]
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 4 or box_h <= 4:
        return None

    box_area = box_w * box_h
    margin_x = int(box_w * 0.35)
    margin_y = int(box_h * 0.35)
    cx1, cy1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    cx2, cy2 = min(w, x2 + margin_x), min(h, y2 + margin_y)
    crop = image_bgr[cy1:cy2, cx1:cx2]
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 40, 40)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Centro da caixa original, em coordenadas do crop — usado pra preferir o
    # contorno mais central quando mais de um candidato plausível aparece.
    orig_cx, orig_cy = (x1 + x2) / 2 - cx1, (y1 + y2) / 2 - cy1

    best = None
    best_score = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < box_area * 0.45 or area > box_area * 2.2:
            continue
        peri = cv2.arcLength(contour, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        rect = cv2.minAreaRect(contour)
        (rw, rh) = rect[1]
        if rw <= 0 or rh <= 0:
            continue
        ratio = max(rw, rh) / min(rw, rh)
        if not (1.0 <= ratio <= 2.4):
            continue

        rcx, rcy = rect[0]
        dist = ((rcx - orig_cx) ** 2 + (rcy - orig_cy) ** 2) ** 0.5
        # Prioriza proximidade do centro original; desempate por área mais
        # próxima da área da caixa detectada (evita pegar um contorno bem
        # maior/menor que "engoliu" ou "perdeu" parte da placa).
        area_penalty = abs(area - box_area) / box_area
        score = dist + area_penalty * max(box_w, box_h) * 0.5
        if best_score is None or score < best_score:
            best_score = score
            best = rect

    if best is None:
        return None

    box_pts = cv2.boxPoints(best)
    box_pts[:, 0] += cx1
    box_pts[:, 1] += cy1
    return _order_box_points(box_pts)


def _blue_fraction(image_bgr: np.ndarray, poly: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
    if w <= 0 or h <= 0:
        return 0.0
    x2, y2 = x + w, y + h
    ih, iw = image_bgr.shape[:2]
    x, y = max(0, x), max(0, y)
    x2, y2 = min(iw, x2), min(ih, y2)
    if x2 <= x or y2 <= y:
        return 0.0

    roi = image_bgr[y:y2, x:x2]
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    shifted = (poly - np.array([x, y])).astype(np.int32)
    cv2.fillConvexPoly(mask, shifted, 255)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lo = np.array([_BLUE_HUE_RANGE[0], _BLUE_SAT_MIN, _BLUE_VAL_MIN])
    hi = np.array([_BLUE_HUE_RANGE[1], 255, 255])
    blue_mask = cv2.inRange(hsv, lo, hi)

    region_px = cv2.countNonZero(mask)
    if region_px == 0:
        return 0.0
    blue_px = cv2.countNonZero(cv2.bitwise_and(blue_mask, mask))
    return blue_px / region_px


def _split_quad(quad: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Corta o retângulo `quad` (4 pontos cíclicos; p0-p1 e p2-p3 devem ser o
    par de lados COMPRIDOS) perpendicularmente ao lado comprido, na fração
    `t` medida a partir do lado p0-p1. Cada pedaço resultante mantém o
    comprimento TOTAL do lado comprido e fica com uma fração do lado curto —
    é assim que uma faixa de cabeçalho (que atravessa a placa inteira de
    largura, ocupando só uma fatia da altura) tem que ser cortada.

    Devolve (parte_perto_de_p0p1, parte_perto_de_p2p3), cada uma já na ordem
    canônica [ponto_de_corte, canto_real, canto_real, ponto_de_corte] — a
    borda entre o último e o primeiro ponto (wraparound) é sempre a linha de
    corte; a borda do meio é sempre a borda real da placa naquela ponta. Isso
    deixa `pad_body_polygon` livre de se importar com qual lado foi escolhido
    como corpo.
    """
    p0, p1, p2, p3 = quad
    cut_near_01 = p0 + t * (p3 - p0)
    cut_near_12 = p1 + t * (p2 - p1)
    part_near_01 = np.array([cut_near_01, p0, p1, cut_near_12], dtype=np.float32)
    part_near_23 = np.array([cut_near_12, p2, p3, cut_near_01], dtype=np.float32)
    return part_near_01, part_near_23


def _quad_band(quad: np.ndarray, t_lo: float, t_hi: float) -> np.ndarray:
    """Faixa fina do quad entre as frações `t_lo` e `t_hi` ao longo do eixo
    curto (mesma convenção de `_split_quad`: p0-p1/p2-p3 são o par comprido).
    Ao contrário de `_split_quad`, isso não é cumulativo a partir da borda —
    é só o intervalo [t_lo, t_hi], o que é o que precisamos pra medir
    densidade de azul numa fatia sem diluir com o que já foi visto antes.
    """
    p0, p1, p2, p3 = quad
    a_lo = p0 + t_lo * (p3 - p0)
    a_hi = p0 + t_hi * (p3 - p0)
    b_lo = p1 + t_lo * (p2 - p1)
    b_hi = p1 + t_hi * (p2 - p1)
    return np.array([a_lo, b_lo, b_hi, a_hi], dtype=np.float32)


def find_header_cut(image_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """Se uma faixa de cabeçalho azul (padrão Mercosul) for identificada com
    confiança numa das pontas do lado curto de `quad`, devolve o polígono do
    "corpo" da placa (a parte com letras/números, que deve ser borrada) sem a
    faixa de cabeçalho. Devolve None se não achar cabeçalho com confiança —
    quem chama deve borrar o quad inteiro nesse caso.
    """
    rect = cv2.minAreaRect(quad.astype(np.float32))
    box_pts = cv2.boxPoints(rect)

    # Qual dos 4 lados é "comprido" (geometricamente) fica ambíguo demais pra
    # placas quase quadradas (comum em moto) — minAreaRect pode devolver w/h
    # trocados por causa de arredondamento, não por causa da placa em si.
    # Em vez de decidir isso pela geometria, testamos os 4 lados diretamente
    # pelo conteúdo: o lado do cabeçalho é o único onde uma faixa fina "colada"
    # nele cobre quase toda a extensão do lado com azul — os dois lados
    # vizinhos pegam só a pontinha do cabeçalho (diluído no resto da faixa,
    # que é altura/largura da placa) e o lado oposto não pega nada.
    probe_frac = 0.16
    best_frac, best_pts = -1.0, box_pts
    for i in range(4):
        rolled = np.roll(box_pts, -i, axis=0)
        frac = _blue_fraction(image_bgr, _quad_band(rolled, 0.0, probe_frac))
        if frac > best_frac:
            best_frac, best_pts = frac, rolled

    if best_frac < _HEADER_MIN_BLUE_FRAC:
        return None
    box_pts = best_pts

    # Varre em fatias finas (não cumulativas) a partir da borda do cabeçalho,
    # procurando onde a densidade de azul cai — essa é a borda real do
    # cabeçalho. Uma medição cumulativa (do início até `t`) dilui aos poucos
    # em vez de cair de forma nítida na borda, e super-estima o cabeçalho.
    steps = 24
    step_frac = _HEADER_MAX_FRAC / steps
    boundary_t = probe_frac
    for i in range(steps):
        t_lo, t_hi = i * step_frac, (i + 1) * step_frac
        frac = _blue_fraction(image_bgr, _quad_band(box_pts, t_lo, t_hi))
        if frac < best_frac * 0.35:
            break
        boundary_t = t_hi

    header_t = min(_HEADER_MAX_FRAC, max(0.0, boundary_t - _HEADER_SAFETY_MARGIN_FRAC))
    if header_t <= 0.02:
        return None

    _, body = _split_quad(box_pts, header_t)
    return body


def pad_quad(quad: np.ndarray, padding_ratio: float) -> np.ndarray:
    """Expande um quad uniformemente a partir do centro (equivalente ao padding
    axis-aligned de antes, generalizado pra qualquer retângulo rotacionado).
    Usado quando não há corte de cabeçalho (a placa inteira vai ser borrada, e
    não existe uma borda "protegida" a preservar)."""
    center = quad.mean(axis=0)
    return (quad - center) * (1 + padding_ratio) + center


def pad_body_polygon(body: np.ndarray, padding_ratio: float) -> np.ndarray:
    """Expande o polígono do "corpo" (devolvido por `find_header_cut`) só nas
    3 bordas que são a borda real da placa — nunca na borda do corte, que é
    onde o cabeçalho preservado começa. Empurrar essa borda pra fora comeria
    de volta a faixa que o usuário pediu pra manter visível.

    `body` segue sempre a ordem [cut_a, corner_1, corner_2, cut_b] devolvida
    por `_split_quad` (dois pontos de corte, dois cantos reais da placa).
    """
    cut_a, corner_1, corner_2, cut_b = body

    # Direção "pra longe do corte": da linha de corte até a borda real —
    # nessa direção, corner_1/corner_2 (a borda real) avançam pra fora;
    # cut_a/cut_b (a borda do corte, vizinha ao cabeçalho) não se mexem nela.
    away_dir = (corner_1 + corner_2) / 2 - (cut_a + cut_b) / 2
    away_len = float(np.linalg.norm(away_dir))
    away_dir = away_dir / away_len if away_len > 1e-6 else np.zeros(2, dtype=np.float32)

    # Direção lateral: ao longo da própria linha de corte (== ao longo da
    # borda real corner_1-corner_2), não da aresta lateral cut_a-corner_1
    # (essa aresta lateral aponta na mesma direção de `away_dir`, não seria
    # perpendicular a ela — usar ela aqui empurraria a linha de corte pra
    # dentro do cabeçalho de um lado, o bug original desta função).
    side_vec = corner_1 - corner_2
    side_len = float(np.linalg.norm(side_vec))
    side_dir = side_vec / side_len if side_len > 1e-6 else np.zeros(2, dtype=np.float32)

    pad_away = away_len * padding_ratio
    pad_side = side_len * padding_ratio

    new_cut_a = cut_a + side_dir * pad_side
    new_cut_b = cut_b - side_dir * pad_side
    new_corner_1 = corner_1 + side_dir * pad_side + away_dir * pad_away
    new_corner_2 = corner_2 - side_dir * pad_side + away_dir * pad_away

    return np.array([new_cut_a, new_corner_1, new_corner_2, new_cut_b], dtype=np.float32)
