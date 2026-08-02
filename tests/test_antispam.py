from datetime import datetime, timedelta, timezone

from app import db
from app.config import settings
from app.presence import PresenceManager
from app.tracker import Tracker

from .helpers import make_detection, process_frame


class _NullRecognizer:
    def match_unknown_cluster(self, conn, embedding):
        return None, 0.0


def test_same_known_face_600_frames_yields_exactly_one_notification(conn):
    """Spec 1 y 12.1: una persona parada 10 minutos genera exactamente 1 notificacion."""
    person_id = db.create_person(conn, "Juan Perez", is_known=True, notify=True)

    tracker = Tracker(settings.IOU_THRESHOLD, settings.TRACK_MAX_AGE)
    presence = PresenceManager()
    notify_events: list = []
    recognizer = _NullRecognizer()
    now = datetime.now(timezone.utc)

    bbox = (100, 100, 200, 200)
    for i in range(600):
        det = make_detection(bbox, matched_person_id=person_id, match_score=0.85)
        process_frame(conn, tracker, presence, recognizer, [det], now, notify_events)
        now += timedelta(milliseconds=500)

    assert len(notify_events) == 1
    assert notify_events[0]["person_id"] == person_id
    assert notify_events[0]["is_known"] is True
