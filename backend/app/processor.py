"""Processamento das fotos: detecção + redação + geração de thumbnails, e execução em lote."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from . import config, database as db, detector, imaging

logger = logging.getLogger("placadetect.processor")

_run_lock = threading.Lock()
_state_lock = threading.Lock()
_batch_state: dict[str, Any] = {
    "active": False,
    "total": 0,
    "done": 0,
    "cancel_requested": False,
    "started_at": None,
    "finished_at": None,
}


def get_batch_state() -> dict[str, Any]:
    with _state_lock:
        return dict(_batch_state)


def request_cancel() -> None:
    with _state_lock:
        _batch_state["cancel_requested"] = True


def process_one(photo_id: int, settings: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Processa uma única foto (detecção + redação) e grava o resultado no banco."""
    settings = settings or {}
    model_name = settings.get("model_name", config.DEFAULT_MODEL_NAME)
    conf_thresh = settings.get("conf_thresh", config.DEFAULT_CONF_THRESH)
    style = settings.get("redaction_style", config.DEFAULT_REDACTION_STYLE)

    photo = db.get_photo(photo_id)
    if not photo:
        raise ValueError(f"foto {photo_id} não encontrada")

    db.set_processing(photo_id)
    input_path = Path(photo["input_path"])

    try:
        image = imaging.load_image_bgr(input_path)
        h, w = image.shape[:2]

        detections = detector.detect_plates(image, model_name=model_name, conf_thresh=conf_thresh)

        output_path = config.OUTPUT_DIR / f"{photo_id}_{input_path.stem}.jpg"
        thumb_out_path = config.THUMBS_DIR / f"{photo_id}_out.jpg"
        thumb_in_path = config.THUMBS_DIR / f"{photo_id}_in.jpg"

        if detections:
            redacted = imaging.redact_plates(image, detections, style=style)
            status = "success"
        else:
            redacted = image
            status = "no_plate"

        imaging.save_image_bgr(redacted, output_path)
        imaging.make_thumbnail(redacted, thumb_out_path)
        imaging.make_thumbnail(image, thumb_in_path)

        db.set_result(
            photo_id,
            status=status,
            output_path=str(output_path),
            thumb_input_path=str(thumb_in_path),
            thumb_output_path=str(thumb_out_path),
            num_plates=len(detections),
            detections=[d.to_dict() for d in detections],
            width=w,
            height=h,
        )
        return db.get_photo(photo_id)
    except Exception as exc:  # noqa: BLE001 - queremos capturar e registrar qualquer falha por foto
        logger.exception("Falha ao processar foto %s (%s)", photo_id, input_path)
        db.set_result(photo_id, status="error", error_message=str(exc))
        return db.get_photo(photo_id)


def run_batch(photo_ids: list[int], settings: Optional[dict[str, Any]] = None) -> None:
    """Executa o processamento de um lote em background, atualizando o progresso."""
    if not _run_lock.acquire(blocking=False):
        logger.warning("Tentativa de iniciar um lote enquanto outro está em execução; ignorado.")
        return

    with _state_lock:
        _batch_state.update(
            active=True,
            total=len(photo_ids),
            done=0,
            cancel_requested=False,
            started_at=time.time(),
            finished_at=None,
        )

    try:
        detector.warmup(
            settings.get("model_name") if settings else None,
            settings.get("conf_thresh") if settings else None,
        )
        for photo_id in photo_ids:
            with _state_lock:
                if _batch_state["cancel_requested"]:
                    break
            process_one(photo_id, settings)
            with _state_lock:
                _batch_state["done"] += 1
    finally:
        with _state_lock:
            _batch_state["active"] = False
            _batch_state["finished_at"] = time.time()
        _run_lock.release()


def start_batch_async(photo_ids: list[int], settings: Optional[dict[str, Any]] = None) -> bool:
    if get_batch_state()["active"]:
        return False
    thread = threading.Thread(target=run_batch, args=(photo_ids, settings), daemon=True)
    thread.start()
    return True
