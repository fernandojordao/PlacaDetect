"""Testes do suporte a RAW da Canon (.CR3).

Não há como testar a decodificação de verdade aqui: um .CR3 real (mesmo um
"pequeno" de teste) tem vários MB e este ambiente de sandbox não tem acesso
de rede a nenhum acervo de amostras RAW para baixar um. O que dá pra travar
sem isso: a extensão é reconhecida e roteada pro decoder certo (`rawpy`, não
o Pillow, que não entende CR3), e a conversão RGB→BGR do resultado do
`rawpy` está correta — usando um mock do `rawpy.imread` no lugar de um
arquivo de verdade.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app import config, imaging  # noqa: E402


def test_cr3_extension_is_recognized():
    assert ".cr3" in config.RAW_EXTENSIONS
    assert ".cr3" in config.IMAGE_EXTENSIONS


def test_load_image_bgr_routes_cr3_to_rawpy(tmp_path):
    fake_path = tmp_path / "IMG_1234.CR3"
    fake_path.write_bytes(b"not a real raw file, just needs to exist")

    # RGB sintético (canais bem diferentes, pra confirmar que a ordem foi
    # realmente invertida e não só copiada)
    fake_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    fake_rgb[..., 0] = 10   # R
    fake_rgb[..., 1] = 20   # G
    fake_rgb[..., 2] = 30   # B

    mock_raw = MagicMock()
    mock_raw.postprocess.return_value = fake_rgb
    mock_raw.__enter__.return_value = mock_raw
    mock_raw.__exit__.return_value = False

    with patch("app.imaging.rawpy.imread", return_value=mock_raw) as mock_imread:
        result = imaging.load_image_bgr(fake_path)

    mock_imread.assert_called_once_with(str(fake_path))
    mock_raw.postprocess.assert_called_once_with(use_camera_wb=True, output_bps=8)

    assert result.shape == (4, 4, 3)
    # BGR: canal 0 deve ser o B original (30), canal 2 o R original (10)
    assert result[0, 0, 0] == 30
    assert result[0, 0, 1] == 20
    assert result[0, 0, 2] == 10


def test_load_image_bgr_does_not_use_rawpy_for_jpeg(tmp_path):
    from PIL import Image

    jpeg_path = tmp_path / "foto.jpg"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(jpeg_path)

    with patch("app.imaging.rawpy.imread") as mock_imread:
        imaging.load_image_bgr(jpeg_path)

    mock_imread.assert_not_called()


@pytest.mark.parametrize(
    "original,expected",
    [
        ("foto.jpg", "foto.jpg"),
        ("FOTO.JPEG", "FOTO.JPEG"),
        ("IMG_1234.CR3", "IMG_1234.jpg"),
        ("imagem.png", "imagem.jpg"),
        ("sem_extensao", "sem_extensao.jpg"),
    ],
)
def test_output_filename_always_jpg_except_when_already_jpeg(original, expected):
    from app.main import _output_filename

    assert _output_filename(original) == expected
