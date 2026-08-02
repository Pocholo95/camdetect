import logging
import os
from datetime import datetime, timedelta, timezone

from app import db
from app.config import settings
from app.recognizer import Recognizer

logger = logging.getLogger("face_presence.unknowns")


def get_or_create_cluster(conn, recognizer: Recognizer, embedding) -> tuple[int, bool]:
    """Devuelve (person_id, created) para el cluster de desconocido de este embedding."""
    cluster_id, _score = recognizer.match_unknown_cluster(conn, embedding)
    if cluster_id is not None:
        return cluster_id, False

    # No hay cluster reciente: crear persona nueva is_known=0
    idx = 1
    while db.get_person_by_name(conn, f"Unknown_{idx:03d}") is not None:
        idx += 1
    name = f"Unknown_{idx:03d}"
    person_id = db.create_person(conn, name=name, is_known=False, notify=True)
    logger.info("nuevo cluster de desconocido creado: %s (id=%d)", name, person_id)
    return person_id, True


def record_sighting(conn, cluster_id: int, embedding, face_crop_path: str,
                     full_frame_path: str, quality: float) -> int:
    return db.add_unknown_sighting(
        conn, cluster_id, embedding, face_crop_path, full_frame_path, quality, db.now_iso()
    )


def delete_cluster(conn, cluster_id: int) -> None:
    """Borra un cluster de desconocido por completo: imagenes + fila de persona
    (cascada a embeddings/unknown_sightings)."""
    sightings = db.cluster_embeddings(conn, cluster_id)
    for s in sightings:
        for path in (s["face_crop"], s["full_frame"]):
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass
    db.delete_person(conn, cluster_id)


def cleanup_stale_clusters(conn, media_dir: str) -> int:
    """Borra clusters desconocidos sin revisar mas antiguos que UNKNOWN_RETENTION_DAYS."""
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=settings.UNKNOWN_RETENTION_DAYS)
    ).isoformat()

    stale_rows = conn.execute(
        "SELECT id, face_crop, full_frame FROM unknown_sightings "
        "WHERE reviewed = 0 AND seen_at < ?",
        (cutoff,),
    ).fetchall()

    for row in stale_rows:
        for path in (row["face_crop"], row["full_frame"]):
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass

    cluster_ids = db.delete_stale_unknown_clusters(conn, cutoff)
    if cluster_ids:
        logger.info("limpieza: %d clusters desconocidos eliminados", len(cluster_ids))
    return len(cluster_ids)
