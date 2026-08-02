import asyncio
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app import db, unknowns
from app.config import settings
from app.notifier import NotificationEvent, Notifier
from app.presence import PresenceManager
from app.quality import evaluate_face
from app.recognizer import Recognizer, l2_normalize
from app.tracker import Detection, Track, Tracker

logger = logging.getLogger("face_presence.capture")


class PipelineStats:
    def __init__(self):
        self.frames_read = 0
        self.frames_processed = 0
        self.frames_discarded_fps = 0
        self.faces_detected = 0
        self.faces_rejected = 0
        self.notifications_sent = 0
        self.last_frame_ts: float | None = None
        self.real_fps = 0.0
        self.reconnects = 0
        self.started_at = time.time()

    def as_dict(self) -> dict:
        return {
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "frames_discarded_fps": self.frames_discarded_fps,
            "faces_detected": self.faces_detected,
            "faces_rejected": self.faces_rejected,
            "notifications_sent": self.notifications_sent,
            "real_fps": round(self.real_fps, 2),
            "reconnects": self.reconnects,
            "uptime_sec": round(time.time() - self.started_at, 1),
        }


def draw_annotated_frame(frame: np.ndarray, bbox: tuple, label: str) -> np.ndarray:
    annotated = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = (0, 200, 0)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        annotated, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )
    return annotated


def save_snapshot(frame_bgr: np.ndarray, media_dir: str, prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    subdir = Path(media_dir) / "snapshots"
    subdir.mkdir(parents=True, exist_ok=True)
    path = subdir / f"{prefix}_{ts}.jpg"
    cv2.imwrite(str(path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return str(path)


def save_crop(face_bgr: np.ndarray, media_dir: str, prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    subdir = Path(media_dir) / "crops"
    subdir.mkdir(parents=True, exist_ok=True)
    path = subdir / f"{prefix}_{ts}.jpg"
    cv2.imwrite(str(path), face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return str(path)


class RTSPReader:
    """Lee el stream RTSP, descarta frames atrasados y reconecta con backoff."""

    def __init__(self, url: str):
        self.url = url
        self.cap: cv2.VideoCapture | None = None
        self._backoff = 1.0

    def _open(self) -> bool:
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self.cap.isOpened()

    def read_latest(self, stats: PipelineStats) -> np.ndarray | None:
        """Devuelve el frame mas reciente disponible, descartando el backlog."""
        if self.cap is None or not self.cap.isOpened():
            if not self._open():
                time.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, 30.0)
                stats.reconnects += 1
                return None
            self._backoff = 1.0

        ok, frame = self.cap.read()
        if not ok or frame is None:
            logger.warning("lectura RTSP fallida, reconectando")
            self.cap.release()
            self.cap = None
            time.sleep(min(self._backoff, 5.0))
            self._backoff = min(self._backoff * 2, 30.0)
            stats.reconnects += 1
            return None

        stats.frames_read += 1
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class CaptureWorker:
    """Hilo de captura + pipeline completo: deteccion, tracking, matching, notificaciones."""

    def __init__(self, loop: asyncio.AbstractEventLoop, notifier: Notifier):
        self.loop = loop
        self.notifier = notifier
        self.stats = PipelineStats()
        self.recognizer = Recognizer()
        self.tracker = Tracker(settings.IOU_THRESHOLD, settings.TRACK_MAX_AGE)
        self.presence = PresenceManager()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_jpeg: bytes | None = None
        self._jpeg_lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="capture-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self._conn is not None:
            self.presence.shutdown(self._conn)
            self._conn.close()

    def latest_jpeg(self) -> bytes | None:
        with self._jpeg_lock:
            return self._latest_jpeg

    def _set_latest_jpeg(self, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._jpeg_lock:
                self._latest_jpeg = buf.tobytes()

    def _schedule_notify(self, event: NotificationEvent) -> None:
        asyncio.run_coroutine_threadsafe(
            self.notifier.notify(self._conn, event), self.loop
        )
        self.stats.notifications_sent += 1

    def _run(self) -> None:
        self._conn = db.get_connection()
        db.init_db(self._conn)
        reader = RTSPReader(settings.RTSP_URL)

        min_interval = 1.0 / settings.TARGET_FPS if settings.TARGET_FPS > 0 else 0.0
        last_processed_at = 0.0
        fps_window: list[float] = []

        while not self._stop_event.is_set():
            frame = reader.read_latest(self.stats)
            if frame is None:
                continue

            now = time.monotonic()
            if now - last_processed_at < min_interval:
                self.stats.frames_discarded_fps += 1
                continue
            last_processed_at = now

            try:
                self._process_frame(frame)
            except Exception:
                logger.exception("error procesando frame")

            fps_window.append(now)
            fps_window = [t for t in fps_window if now - t < 5.0]
            if len(fps_window) >= 2:
                self.stats.real_fps = len(fps_window) / (fps_window[-1] - fps_window[0] + 1e-6)

        reader.release()

    def _process_frame(self, frame: np.ndarray) -> None:
        conn = self._conn
        self.stats.frames_processed += 1
        faces = self.recognizer.get_faces(frame)
        self.stats.faces_detected += len(faces)

        detections: list[Detection] = []
        for face in faces:
            accepted, quality, reason = evaluate_face(frame, face.bbox, face.det_score, face.kps)
            if not accepted:
                self.stats.faces_rejected += 1
                continue

            embedding = l2_normalize(face.embedding)
            person_id, score = self.recognizer.match_known(conn, embedding)

            x1, y1, x2, y2 = [int(v) for v in face.bbox[:4]]
            h, w = frame.shape[:2]
            face_crop = frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)].copy()

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                quality=quality,
                face_crop=face_crop,
                full_frame=frame.copy(),
                embedding=embedding,
                matched_person_id=person_id,
                match_score=score if person_id is not None else 0.0,
            ))

        updated_tracks, dead_tracks = self.tracker.step(detections)

        now_dt = datetime.now(timezone.utc)
        annotated_preview = frame

        for track in updated_tracks:
            # track.confirmed_identity ya trae la resolucion por votos de personas
            # conocidas (tracker.step). Si no hay ganador conocido, el track se
            # trata como desconocido y se le asigna (una sola vez) un cluster
            # estable para que reciba el mismo cooldown que una persona conocida.
            if track.confirmed_identity is None:
                if track.unknown_cluster_id is None:
                    cluster_id, created = unknowns.get_or_create_cluster(
                        conn, self.recognizer, track.embeddings[-1]
                    )
                    if created:
                        crop_path = save_crop(track.best_face_crop, settings.MEDIA_DIR, "unk")
                        frame_path = save_snapshot(
                            track.best_full_frame, settings.MEDIA_DIR, "unk"
                        )
                        unknowns.record_sighting(
                            conn, cluster_id, track.embeddings[-1], crop_path, frame_path,
                            track.best_quality,
                        )
                    track.unknown_cluster_id = cluster_id
                track.confirmed_identity = track.unknown_cluster_id

            identity = track.confirmed_identity
            if identity is None:
                continue

            person_row = db.get_person(conn, identity)
            if person_row is None:
                continue

            score = track.best_identity_score()
            label = f"{person_row['name']} ({score:.2f})"
            annotated = draw_annotated_frame(track.best_full_frame, track.bbox, label)
            annotated_preview = draw_annotated_frame(annotated_preview, track.bbox, label)

            snapshot_path = save_snapshot(annotated, settings.MEDIA_DIR, f"p{identity}")
            log_id = self.presence.update(conn, identity, now_dt, score, snapshot_path)
            track.log_id = log_id

            if (
                not track.notified
                and track.n_frames >= settings.MIN_FRAMES_CONFIRM
                and bool(person_row["notify"])
            ):
                event = NotificationEvent(
                    person_id=identity,
                    name=person_row["name"],
                    is_known=bool(person_row["is_known"]),
                    score=score,
                    snapshot_path=snapshot_path,
                    timestamp=now_dt.isoformat(),
                )
                self._schedule_notify(event)
                track.notified = True

        for track in dead_tracks:
            self._archive_dead_track(conn, track)

        self._set_latest_jpeg(annotated_preview)

    def _archive_dead_track(self, conn: sqlite3.Connection, track: Track) -> None:
        if track.confirmed_identity is None or track.best_face_crop is None:
            return
        person_row = db.get_person(conn, track.confirmed_identity)
        if person_row is None or bool(person_row["is_known"]):
            return
        crop_path = save_crop(track.best_face_crop, settings.MEDIA_DIR, "unk_best")
        frame_path = save_snapshot(track.best_full_frame, settings.MEDIA_DIR, "unk_best")
        unknowns.record_sighting(
            conn, track.confirmed_identity, track.embeddings[-1], crop_path, frame_path,
            track.best_quality,
        )
