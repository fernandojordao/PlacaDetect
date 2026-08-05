"""Ponto de entrada do PlacaDetect: sobe a API + interface web em http://localhost:8000"""

import sys
import webbrowser
from pathlib import Path
from threading import Timer

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

HOST = "127.0.0.1"
PORT = 8000


def open_browser() -> None:
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    Timer(1.5, open_browser).start()
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)
