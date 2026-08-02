"""Capa de acceso a SQLite. Guarda embeddings como blobs de float32."""
import sqlite3
import time
import numpy as np
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS known_faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    embedding BLOB NOT NULL,
    crop_path TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    embedding BLOB NOT NULL,
    crop_path TEXT NOT NULL,
    first_seen REAL NOT NULL,
    cluster_id INTEGER,
    FOREIGN KEY (cluster_id) REFERENCES clusters(id)
);

CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | labeled | discarded
    label TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    person TEXT NOT NULL,          -- nombre o 'desconocido'
    confidence REAL,
    snapshot_path TEXT
);
"""


def emb_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_emb(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class Database:
    def __init__(self, path: str):
        self.path = path
        with self._conn() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self.path, timeout=30)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # ---------- known_faces ----------
    def add_known_face(self, name: str, embedding: np.ndarray, crop_path: str = None):
        with self._conn() as con:
            con.execute(
                "INSERT INTO known_faces (name, embedding, crop_path, created_at) VALUES (?,?,?,?)",
                (name, emb_to_blob(embedding), crop_path, time.time()),
            )

    def get_all_known_faces(self):
        with self._conn() as con:
            rows = con.execute("SELECT id, name, embedding FROM known_faces").fetchall()
        return [(r[0], r[1], blob_to_emb(r[2])) for r in rows]

    # ---------- pending_faces ----------
    def add_pending_face(self, embedding: np.ndarray, crop_path: str):
        with self._conn() as con:
            cur = con.execute(
                "INSERT INTO pending_faces (embedding, crop_path, first_seen) VALUES (?,?,?)",
                (emb_to_blob(embedding), crop_path, time.time()),
            )
            return cur.lastrowid

    def get_recent_pending_embeddings(self, limit: int):
        with self._conn() as con:
            rows = con.execute(
                "SELECT embedding FROM pending_faces ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [blob_to_emb(r[0]) for r in rows]

    def get_unclustered_pending(self):
        with self._conn() as con:
            rows = con.execute(
                "SELECT id, embedding, crop_path FROM pending_faces WHERE cluster_id IS NULL"
            ).fetchall()
        return [(r[0], blob_to_emb(r[1]), r[2]) for r in rows]

    def assign_cluster(self, pending_id: int, cluster_id: int):
        with self._conn() as con:
            con.execute(
                "UPDATE pending_faces SET cluster_id=? WHERE id=?", (cluster_id, pending_id)
            )

    # ---------- clusters ----------
    def create_cluster(self) -> int:
        with self._conn() as con:
            cur = con.execute(
                "INSERT INTO clusters (status, created_at) VALUES ('pending', ?)", (time.time(),)
            )
            return cur.lastrowid

    def get_clusters(self, status: str = "pending"):
        with self._conn() as con:
            rows = con.execute(
                "SELECT id, status, label, created_at FROM clusters WHERE status=? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        return rows

    def get_cluster_faces(self, cluster_id: int):
        with self._conn() as con:
            rows = con.execute(
                "SELECT id, embedding, crop_path FROM pending_faces WHERE cluster_id=?",
                (cluster_id,),
            ).fetchall()
        return [(r[0], blob_to_emb(r[1]), r[2]) for r in rows]

    def label_cluster(self, cluster_id: int, name: str):
        with self._conn() as con:
            con.execute(
                "UPDATE clusters SET status='labeled', label=? WHERE id=?", (name, cluster_id)
            )

    def discard_cluster(self, cluster_id: int):
        with self._conn() as con:
            con.execute("UPDATE clusters SET status='discarded' WHERE id=?", (cluster_id,))

    def delete_pending_faces(self, pending_ids):
        with self._conn() as con:
            con.executemany("DELETE FROM pending_faces WHERE id=?", [(i,) for i in pending_ids])

    # ---------- events ----------
    def add_event(self, person: str, confidence: float, snapshot_path: str = None):
        with self._conn() as con:
            con.execute(
                "INSERT INTO events (timestamp, person, confidence, snapshot_path) VALUES (?,?,?,?)",
                (time.time(), person, confidence, snapshot_path),
            )
