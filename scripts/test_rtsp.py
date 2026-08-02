"""Verifica que el stream RTSP se puede leer antes de arrancar el resto del sistema.

Uso:
    python scripts/test_rtsp.py [rtsp_url]

Si no se pasa URL, usa RTSP_URL de .env / variables de entorno.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from app.config import settings


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else settings.RTSP_URL
    print(f"Probando RTSP: {url}")

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: no se pudo abrir el stream. Revisa URL, credenciales y red.")
        sys.exit(1)

    n_ok = 0
    n_fail = 0
    start = time.time()
    print("Leyendo 30 frames de prueba...")
    for i in range(30):
        ok, frame = cap.read()
        if ok and frame is not None:
            n_ok += 1
        else:
            n_fail += 1
        time.sleep(0.05)

    elapsed = time.time() - start
    cap.release()

    print(f"Frames OK: {n_ok}/30, fallidos: {n_fail}/30, en {elapsed:.1f}s")
    if n_ok == 0:
        print("ERROR: no se pudo leer ningun frame. El stream no es utilizable.")
        sys.exit(1)

    print("OK: el stream RTSP se lee correctamente.")
    if frame is not None:
        h, w = frame.shape[:2]
        print(f"Resolucion detectada: {w}x{h}")


if __name__ == "__main__":
    main()
