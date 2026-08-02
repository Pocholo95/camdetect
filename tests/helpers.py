"""Harness minimo que reproduce la logica de decision de capture.py
(identidad de track -> cluster de desconocido -> evento de notificacion)
sin depender de OpenCV/InsightFace, para poder probar las reglas anti-spam
de la seccion 1 y 12 de SPEC.md de forma rapida y determinista.
"""
from datetime import datetime

from app import db, unknowns
from app.config import settings
from app.presence import PresenceManager
from app.tracker import Detection, Tracker


def make_detection(bbox, matched_person_id=None, match_score=0.0, quality=0.8, embedding=None):
    import numpy as np
    if embedding is None:
        embedding = np.random.default_rng(0).normal(size=512).astype("float32")
        embedding = embedding / np.linalg.norm(embedding)
    dummy_img = np.zeros((10, 10, 3), dtype="uint8")
    return Detection(
        bbox=bbox,
        quality=quality,
        face_crop=dummy_img,
        full_frame=dummy_img,
        embedding=embedding,
        matched_person_id=matched_person_id,
        match_score=match_score,
    )


def process_frame(conn, tracker: Tracker, presence: PresenceManager, recognizer,
                   detections: list[Detection], now: datetime, notify_events: list):
    updated_tracks, dead_tracks = tracker.step(detections)

    for track in updated_tracks:
        if track.confirmed_identity is None:
            if track.unknown_cluster_id is None:
                cluster_id, created = unknowns.get_or_create_cluster(
                    conn, recognizer, track.embeddings[-1]
                )
                if created:
                    unknowns.record_sighting(
                        conn, cluster_id, track.embeddings[-1], "crop.jpg", "frame.jpg",
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
        presence.update(conn, identity, now, score, None)

        if (
            not track.notified
            and track.n_frames >= settings.MIN_FRAMES_CONFIRM
            and bool(person_row["notify"])
        ):
            notify_events.append({
                "person_id": identity,
                "name": person_row["name"],
                "is_known": bool(person_row["is_known"]),
            })
            track.notified = True

    return updated_tracks, dead_tracks
