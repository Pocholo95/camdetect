import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    is_known    INTEGER NOT NULL DEFAULT 1,
    notify      INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY,
    person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    vector      BLOB NOT NULL,
    condition   TEXT,
    source      TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_emb_person ON embeddings(person_id);

CREATE TABLE IF NOT EXISTS presence_logs (
    id          INTEGER PRIMARY KEY,
    person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL,
    n_frames    INTEGER NOT NULL DEFAULT 1,
    best_score  REAL,
    snapshot    TEXT
);
CREATE INDEX IF NOT EXISTS idx_logs_person_time ON presence_logs(person_id, first_seen);

CREATE TABLE IF NOT EXISTS unknown_sightings (
    id          INTEGER PRIMARY KEY,
    cluster_id  INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    vector      BLOB NOT NULL,
    face_crop   TEXT NOT NULL,
    full_frame  TEXT NOT NULL,
    quality     REAL NOT NULL,
    seen_at     TEXT NOT NULL,
    reviewed    INTEGER NOT NULL DEFAULT 0
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    Path(settings.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def get_cursor(conn: sqlite3.Connection):
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def vector_to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


# --- Persons -----------------------------------------------------------

def create_person(conn, name: str, is_known: bool = True, notify: bool = True) -> int:
    with get_cursor(conn) as cur:
        cur.execute(
            "INSERT INTO persons (name, is_known, notify, created_at) VALUES (?, ?, ?, ?)",
            (name, int(is_known), int(notify), now_iso()),
        )
        return cur.lastrowid


def get_person(conn, person_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()


def get_person_by_name(conn, name: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM persons WHERE name = ?", (name,)).fetchone()


def list_persons(conn, is_known: bool | None = None) -> list[sqlite3.Row]:
    if is_known is None:
        return conn.execute("SELECT * FROM persons ORDER BY name").fetchall()
    return conn.execute(
        "SELECT * FROM persons WHERE is_known = ? ORDER BY name", (int(is_known),)
    ).fetchall()


def set_person_notify(conn, person_id: int, notify: bool) -> None:
    with get_cursor(conn) as cur:
        cur.execute("UPDATE persons SET notify = ? WHERE id = ?", (int(notify), person_id))


def delete_person(conn, person_id: int) -> None:
    with get_cursor(conn) as cur:
        cur.execute("DELETE FROM persons WHERE id = ?", (person_id,))


def promote_unknown_to_named(conn, person_id: int, name: str) -> None:
    with get_cursor(conn) as cur:
        cur.execute(
            "UPDATE persons SET name = ?, is_known = 1 WHERE id = ?", (name, person_id)
        )


# --- Embeddings ----------------------------------------------------------

def add_embedding(
    conn, person_id: int, vector: np.ndarray, condition: str | None, source: str
) -> int:
    with get_cursor(conn) as cur:
        cur.execute(
            "INSERT INTO embeddings (person_id, vector, condition, source, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (person_id, vector_to_blob(vector), condition, source, now_iso()),
        )
        return cur.lastrowid


def list_embeddings(conn, person_id: int | None = None) -> list[sqlite3.Row]:
    if person_id is None:
        return conn.execute("SELECT * FROM embeddings").fetchall()
    return conn.execute(
        "SELECT * FROM embeddings WHERE person_id = ?", (person_id,)
    ).fetchall()


def delete_embedding(conn, embedding_id: int) -> None:
    with get_cursor(conn) as cur:
        cur.execute("DELETE FROM embeddings WHERE id = ?", (embedding_id,))


def move_embeddings(conn, from_person_id: int, to_person_id: int) -> None:
    with get_cursor(conn) as cur:
        cur.execute(
            "UPDATE embeddings SET person_id = ? WHERE person_id = ?",
            (to_person_id, from_person_id),
        )


# --- Presence logs ---------------------------------------------------------

def open_presence_log(conn, person_id: int, seen_at: str, best_score: float | None,
                       snapshot: str | None) -> int:
    with get_cursor(conn) as cur:
        cur.execute(
            "INSERT INTO presence_logs (person_id, first_seen, last_seen, n_frames, "
            "best_score, snapshot) VALUES (?, ?, ?, 1, ?, ?)",
            (person_id, seen_at, seen_at, best_score, snapshot),
        )
        return cur.lastrowid


def update_presence_log(conn, log_id: int, last_seen: str, best_score: float | None,
                         snapshot: str | None, n_frames: int) -> None:
    with get_cursor(conn) as cur:
        cur.execute(
            "UPDATE presence_logs SET last_seen = ?, best_score = ?, snapshot = ?, "
            "n_frames = ? WHERE id = ?",
            (last_seen, best_score, snapshot, n_frames, log_id),
        )


def count_sightings_today(conn, person_id: int) -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = conn.execute(
        "SELECT COUNT(*) as c FROM presence_logs WHERE person_id = ? AND first_seen >= ?",
        (person_id, today),
    ).fetchone()
    return row["c"] if row else 0


def list_presence_logs(conn, person_id: int | None = None, date_from: str | None = None,
                        date_to: str | None = None, limit: int = 200) -> list[sqlite3.Row]:
    query = "SELECT * FROM presence_logs WHERE 1=1"
    params: list = []
    if person_id is not None:
        query += " AND person_id = ?"
        params.append(person_id)
    if date_from is not None:
        query += " AND first_seen >= ?"
        params.append(date_from)
    if date_to is not None:
        query += " AND first_seen <= ?"
        params.append(date_to)
    query += " ORDER BY first_seen DESC LIMIT ?"
    params.append(limit)
    return conn.execute(query, params).fetchall()


# --- Unknown sightings ------------------------------------------------------

def add_unknown_sighting(conn, cluster_id: int, vector: np.ndarray, face_crop: str,
                          full_frame: str, quality: float, seen_at: str) -> int:
    with get_cursor(conn) as cur:
        cur.execute(
            "INSERT INTO unknown_sightings (cluster_id, vector, face_crop, full_frame, "
            "quality, seen_at, reviewed) VALUES (?, ?, ?, ?, ?, ?, 0)",
            (cluster_id, vector_to_blob(vector), face_crop, full_frame, quality, seen_at),
        )
        return cur.lastrowid


def recent_unknown_clusters(conn, since_iso: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT DISTINCT p.id, p.name FROM persons p "
        "JOIN unknown_sightings u ON u.cluster_id = p.id "
        "WHERE p.is_known = 0 AND u.seen_at >= ?",
        (since_iso,),
    ).fetchall()


def cluster_embeddings(conn, cluster_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM unknown_sightings WHERE cluster_id = ?", (cluster_id,)
    ).fetchall()


def mark_reviewed(conn, sighting_ids: list[int]) -> None:
    if not sighting_ids:
        return
    with get_cursor(conn) as cur:
        cur.executemany(
            "UPDATE unknown_sightings SET reviewed = 1 WHERE id = ?",
            [(i,) for i in sighting_ids],
        )


def unreviewed_clusters(conn) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT p.id as cluster_id, p.name, MAX(u.quality) as best_quality, "
        "COUNT(u.id) as n_sightings, MAX(u.seen_at) as last_seen "
        "FROM persons p JOIN unknown_sightings u ON u.cluster_id = p.id "
        "WHERE p.is_known = 0 AND u.reviewed = 0 "
        "GROUP BY p.id ORDER BY last_seen DESC"
    ).fetchall()


def delete_stale_unknown_clusters(conn, older_than_iso: str) -> list[int]:
    rows = conn.execute(
        "SELECT DISTINCT cluster_id FROM unknown_sightings "
        "WHERE reviewed = 0 AND seen_at < ?",
        (older_than_iso,),
    ).fetchall()
    cluster_ids = [r["cluster_id"] for r in rows]
    for cid in cluster_ids:
        delete_person(conn, cid)
    return cluster_ids
