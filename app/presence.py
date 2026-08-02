import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app import db
from app.config import settings

logger = logging.getLogger("face_presence.presence")


@dataclass
class _OpenEntry:
    log_id: int
    last_seen: datetime
    n_frames: int


class PresenceManager:
    """Maquina de estados de presencia (spec seccion 3 esquema + notas).

    Si una identidad reaparece dentro de PRESENCE_TIMEOUT_MIN desde su ultimo
    frame visto, se extiende el mismo presence_log. Pasado ese margen, la
    siguiente aparicion abre un registro nuevo (spec 12.5: 2 registros tras
    una ausencia de 6 min con timeout default de 5 min).
    """

    def __init__(self):
        self._open: dict[int, _OpenEntry] = {}

    def _sweep_expired(self, now: datetime) -> None:
        timeout = timedelta(minutes=settings.PRESENCE_TIMEOUT_MIN)
        expired = [
            pid for pid, entry in self._open.items() if now - entry.last_seen > timeout
        ]
        for pid in expired:
            del self._open[pid]

    def update(self, conn, person_id: int, timestamp: datetime, score: float | None,
               snapshot: str | None) -> int:
        """Registra una deteccion confirmada de person_id en este instante.

        Devuelve el log_id (nuevo o reabierto) del presence_log activo.
        """
        self._sweep_expired(timestamp)
        ts_iso = timestamp.isoformat()

        entry = self._open.get(person_id)
        if entry is not None:
            entry.n_frames += 1
            entry.last_seen = timestamp
            db.update_presence_log(conn, entry.log_id, ts_iso, score, snapshot, entry.n_frames)
            return entry.log_id

        log_id = db.open_presence_log(conn, person_id, ts_iso, score, snapshot)
        self._open[person_id] = _OpenEntry(log_id=log_id, last_seen=timestamp, n_frames=1)
        return log_id

    def currently_present(self) -> list[tuple[int, datetime, int]]:
        """Lista (person_id, last_seen, n_frames) de identidades con presencia abierta."""
        return [(pid, e.last_seen, e.n_frames) for pid, e in self._open.items()]

    def shutdown(self, conn) -> None:
        """Cierre limpio: los presence_logs ya reflejan el ultimo estado conocido."""
        if self._open:
            logger.info("cerrando %d presence_logs abiertos por SIGTERM", len(self._open))
        self._open.clear()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
