import logging
import os
from pathlib import Path

import numpy as np

from app import db
from app.config import settings

logger = logging.getLogger("face_presence.recognizer")

os.environ.setdefault("ORT_NUM_THREADS", str(settings.ORT_NUM_THREADS))


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


class Recognizer:
    """Envuelve InsightFace (buffalo_sc) y el matching contra embeddings conocidos."""

    def __init__(self, models_dir: str | None = None, load_model: bool = True):
        self.face_app = None
        if load_model:
            from insightface.app import FaceAnalysis

            models_root = models_dir or str(Path(settings.MEDIA_DIR).parent / "models")
            Path(models_root).mkdir(parents=True, exist_ok=True)

            self.face_app = FaceAnalysis(name="buffalo_sc", root=models_root)
            self.face_app.prepare(ctx_id=-1, det_size=(640, 640))

        self._known_matrix: np.ndarray | None = None
        self._known_person_ids: list[int] = []
        self._known_dirty = True

    def invalidate_known_cache(self) -> None:
        self._known_dirty = True

    def get_faces(self, frame: np.ndarray) -> list:
        """Devuelve las caras detectadas por InsightFace (bbox, det_score, kps, embedding)."""
        if self.face_app is None:
            raise RuntimeError("Recognizer creado con load_model=False: no hay modelo cargado")
        return self.face_app.get(frame)

    def _rebuild_known_cache(self, conn) -> None:
        rows = conn.execute(
            "SELECT e.person_id as person_id, e.vector as vector "
            "FROM embeddings e JOIN persons p ON p.id = e.person_id "
            "WHERE p.is_known = 1"
        ).fetchall()
        if not rows:
            self._known_matrix = None
            self._known_person_ids = []
            self._known_dirty = False
            return
        vectors = [db.blob_to_vector(r["vector"]) for r in rows]
        self._known_matrix = np.stack(vectors)
        self._known_person_ids = [r["person_id"] for r in rows]
        self._known_dirty = False

    def match_known(self, conn, embedding: np.ndarray) -> tuple[int | None, float]:
        """Aplica las reglas de decision de la seccion 6. Devuelve (person_id|None, score)."""
        if self._known_dirty:
            self._rebuild_known_cache(conn)

        if self._known_matrix is None or len(self._known_person_ids) == 0:
            return None, 0.0

        v = l2_normalize(embedding)
        sims = self._known_matrix @ v

        per_person_best: dict[int, float] = {}
        for pid, sim in zip(self._known_person_ids, sims):
            if sim > per_person_best.get(pid, -1.0):
                per_person_best[pid] = float(sim)

        ranked = sorted(per_person_best.items(), key=lambda kv: kv[1], reverse=True)
        best_person_id, best_score = ranked[0]

        if best_score < settings.SIM_THRESHOLD:
            return None, best_score

        if len(ranked) > 1:
            _, second_score = ranked[1]
            if best_score - second_score < settings.MARGIN_THRESHOLD:
                return None, best_score

        return best_person_id, best_score

    def match_unknown_cluster(self, conn, embedding: np.ndarray) -> tuple[int | None, float]:
        """Clustering de desconocidos (seccion 6.1). Devuelve (cluster_person_id|None, score)."""
        since = _hours_ago_iso(settings.UNKNOWN_TTL_HOURS)
        rows = conn.execute(
            "SELECT cluster_id, vector FROM unknown_sightings WHERE seen_at >= ?",
            (since,),
        ).fetchall()
        if not rows:
            return None, 0.0

        v = l2_normalize(embedding)
        per_cluster_best: dict[int, float] = {}
        for row in rows:
            vec = db.blob_to_vector(row["vector"])
            sim = float(vec @ v)
            cid = row["cluster_id"]
            if sim > per_cluster_best.get(cid, -1.0):
                per_cluster_best[cid] = sim

        best_cluster_id = max(per_cluster_best, key=per_cluster_best.get)
        best_score = per_cluster_best[best_cluster_id]

        if best_score >= settings.UNKNOWN_CLUSTER_THRESHOLD:
            return best_cluster_id, best_score
        return None, best_score


def _hours_ago_iso(hours: float) -> str:
    from datetime import datetime, timedelta, timezone

    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
