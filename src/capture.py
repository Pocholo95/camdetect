"""Lector RTSP en un hilo propio. VideoCapture.read() bloquea, asi que lo
aislamos para que el pipeline principal siempre pueda pedir "el ultimo
frame disponible" sin acumular un buffer de frames viejos."""
import threading
import time
import cv2


class RTSPStream:
    def __init__(self, url: str, reconnect_delay: float = 5.0):
        self.url = url
        self.reconnect_delay = reconnect_delay
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._cap = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        # Buffer minimo: no queremos frames viejos acumulados
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self):
        self._cap = self._open()
        while self._running:
            if not self._cap.isOpened():
                print(f"[capture] no se pudo abrir el stream, reintentando en {self.reconnect_delay}s")
                time.sleep(self.reconnect_delay)
                self._cap.release()
                self._cap = self._open()
                continue

            ok, frame = self._cap.read()
            if not ok:
                print("[capture] frame perdido / stream caido, reconectando...")
                self._cap.release()
                time.sleep(self.reconnect_delay)
                self._cap = self._open()
                continue

            with self._lock:
                self._frame = frame

    def read(self):
        """Devuelve el ultimo frame disponible (o None si todavia no hay)."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()
