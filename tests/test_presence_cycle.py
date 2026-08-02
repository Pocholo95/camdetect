from datetime import datetime, timedelta, timezone

from app import db
from app.config import settings
from app.presence import PresenceManager


def test_reappearance_after_gap_creates_new_presence_log(conn):
    """Spec 12.5: cara aparece, desaparece 6 min, reaparece -> 2 registros distintos."""
    assert settings.PRESENCE_TIMEOUT_MIN < 6, "el gap de prueba debe superar el timeout"

    person_id = db.create_person(conn, "Ana", is_known=True)
    presence = PresenceManager()

    t0 = datetime.now(timezone.utc)
    log_id_1 = presence.update(conn, person_id, t0, 0.8, None)

    gap_end = t0 + timedelta(minutes=6)
    log_id_2 = presence.update(conn, person_id, gap_end, 0.75, None)

    assert log_id_1 != log_id_2

    rows = db.list_presence_logs(conn, person_id=person_id)
    assert len(rows) == 2


def test_reappearance_within_timeout_reuses_same_presence_log(conn):
    person_id = db.create_person(conn, "Beto", is_known=True)
    presence = PresenceManager()

    t0 = datetime.now(timezone.utc)
    log_id_1 = presence.update(conn, person_id, t0, 0.8, None)

    soon_after = t0 + timedelta(minutes=1)
    log_id_2 = presence.update(conn, person_id, soon_after, 0.8, None)

    assert log_id_1 == log_id_2
    rows = db.list_presence_logs(conn, person_id=person_id)
    assert len(rows) == 1
    assert rows[0]["n_frames"] == 2
