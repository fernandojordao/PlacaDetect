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


def _long_edge_direction(box_pts: np.ndarray) -> np.ndarray:
    """Vetor unitário na direção do lado mais comprido de um retângulo (4
    pontos cíclicos). Usar a direção de uma aresta em vez do ângulo que
    `cv2.minAreaRect` devolve evita depender da convenção de sinal/faixa
    desse ângulo (muda entre versões do OpenCV) — a direção de uma aresta
    real não tem essa ambiguidade.
    """
    edges = [box_pts[(i + 1) % 4] - box_pts[i] for i in range(4)]
    lens = [float(np.linalg.norm(e)) for e in edges]
    d = edges[int(np.argmax(lens))]
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)


def _find_header_band_rect(image_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    """Acha o retângulo mínimo (rotacionado) da faixa azul Mercosul perto da
    caixa detectada, ou None. A faixa é um alvo mais fácil que a placa
    inteira pra achar por cor: cor bem distintiva, sem se misturar com pneu/
    carenagem/asfalto do jeito que o contorno da placa inteira se mistura.
    """
    h, w = image_bgr.shape[:2]
    box_w, box_h = x2 - x1, y2 - y1
    margin_x, margin_y = int(box_w * 0.4), int(box_h * 0.4)
    cx1, cy1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    cx2, cy2 = min(w, x2 + margin_x), min(h, y2 + margin_y)
    crop = image_bgr[cy1:cy2, cx1:cx2]
    if crop.shape[0] < 6 or crop.shape[1] < 6:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lo = np.array([_BLUE_HUE_RANGE[0], _BLUE_SAT_MIN, _BLUE_VAL_MIN])
    hi = np.array([_BLUE_HUE_RANGE[1], 255, 255])
    mask = cv2.inRange(hsv, lo, hi)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    box_area = box_w * box_h
    best, best_area = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        # a faixa cobre a largura inteira da placa mas só uma fatia da
        # altura — mesmo assim deve ser uma fração razoável da área da
        # caixa (senão é ruído: reflexo, pontinha de algo azul ao fundo).
        if area < box_area * 0.06:
            continue
        rect = cv2.minAreaRect(c)
        (rw, rh) = rect[1]
        if rw <= 0 or rh <= 0:
            continue
        ratio = max(rw, rh) / min(rw, rh)
        if not (1.8 <= ratio <= 15):  # a faixa é bem alongada, não é a placa toda
            continue
        if area > best_area:
            best_area, best = area, rect

    if best is None:
        return None
    pts = cv2.boxPoints(best)
    pts[:, 0] += cx1
    pts[:, 1] += cy1
    return pts


def _quad_from_header_band(x1: int, y1: int, x2: int, y2: int, band_pts: np.ndarray) -> np.ndarray | None:
    """Reconstrói o quadrilátero da placa inteira a partir da inclinação e
    largura precisas da faixa de cabeçalho, combinadas com a área da caixa
    axis-aligned do detector (confiável para o tamanho, não para o ângulo).

    A faixa dá a direção exata do eixo comprido da placa e sua largura real
    W (ela atravessa a placa de ponta a ponta). Falta a altura real H da
    placa — a caixa do detector só dá a extensão *axis-aligned*, que para um
    retângulo girado é sempre maior que as dimensões reais. Resolve H a
    partir da relação entre uma caixa alinhada aos eixos e as dimensões
    reais de um retângulo girado por um ângulo θ:

        bbox_w = W·|cos θ| + H·|sin θ|
        bbox_h = W·|sin θ| + H·|cos θ|

    (a direção da faixa já dá |cos θ|, |sin θ| diretamente, como as
    componentes do vetor unitário — sem precisar do ângulo em si).
    """
    long_dir = _long_edge_direction(band_pts)
    cos_t, sin_t = abs(float(long_dir[0])), abs(float(long_dir[1]))

    band_w, band_h = cv2.minAreaRect(band_pts.astype(np.float32))[1]
    width_est = max(band_w, band_h)

    box_w, box_h = x2 - x1, y2 - y1
    # Duas equações disponíveis (bbox_w = W·cos_t + H·sin_t, bbox_h = W·sin_t
    # + H·cos_t); usa a que tem o divisor maior, mais estável numericamente
    # perto de ângulos próximos de 0°/90° (onde o outro divisor vai a zero).
    if cos_t >= sin_t:
        height_est = (box_h - width_est * sin_t) / cos_t if cos_t > 1e-3 else None
    else:
        height_est = (box_w - width_est * cos_t) / sin_t if sin_t > 1e-3 else None

    if height_est is None or width_est <= 0:
        return None

    ratio = max(width_est, height_est) / max(1.0, min(width_est, height_est))
    if not (1.0 <= ratio <= 2.4) or height_est <= 4:
        return None

    short_dir = np.array([-long_dir[1], long_dir[0]], dtype=np.float32)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = width_est / 2.0, height_est / 2.0
    corners = np.array(
        [
            [cx - long_dir[0] * hw - short_dir[0] * hh, cy - long_dir[1] * hw - short_dir[1] * hh],
            [cx + long_dir[0] * hw - short_dir[0] * hh, cy + long_dir[1] * hw - short_dir[1] * hh],
            [cx + long_dir[0] * hw + short_dir[0] * hh, cy + long_dir[1] * hw + short_dir[1] * hh],
            [cx - long_dir[0] * hw + short_dir[0] * hh, cy - long_dir[1] * hw + short_dir[1] * hh],
        ],
        dtype=np.float32,
    )
    return corners


def find_plate_quad(image_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray | None:
    """Tenta achar o quadrilátero real da placa (rotacionado) dentro/perto da
    caixa detectada. Devolve 4 pontos (float32, ordem cíclica) em coordenadas
    da imagem original, ou None se nada plausível for encontrado.

    Duas estratégias, nessa ordem:
    1. Via faixa de cabeçalho azul (mais robusta — a faixa é um alvo de cor
       muito mais distintivo e isolado que a placa inteira, que se mistura
       com pneu/carenagem/asfalto quando tentamos achar o contorno dela
       direto). Só serve pra placas Mercosul com cabeçalho identificável.
    2. Contorno genérico (Canny + approxPolyDP) como antes — mais frágil,
       mas não depende de ter uma faixa azul.
    """
    h, w = image_bgr.shape[:2]
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 4 or box_h <= 4:
        return None

    band_pts = _find_header_band_rect(image_bgr, x1, y1, x2, y2)
    if band_pts is not None:
        quad = _quad_from_header_band(x1, y1, x2, y2, band_pts)
        if quad is not None:
            return quad

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
    corte; a borda do meio é sempre a borda real da placa naquela ponta.
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
