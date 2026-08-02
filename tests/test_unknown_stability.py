from datetime import datetime, timedelta, timezone

import numpy as np

from app import db
from app.config import settings
from app.presence import PresenceManager
from app.recognizer import Recognizer
from app.tracker import Tracker

from .helpers import make_detection, process_frame


def test_unregistered_face_100_frames_yields_one_cluster_and_one_notification(conn):
    """Spec 12.2: 100 frames de una cara no registrada -> 1 cluster, 1 notificacion."""
    tracker = Tracker(settings.IOU_THRESHOLD, settings.TRACK_MAX_AGE)
    presence = PresenceManager()
    recognizer = Recognizer(load_model=False)
    notify_events: list = []
    now = datetime.now(timezone.utc)

    rng = np.random.default_rng(99)
    fixed_embedding = rng.normal(size=512).astype("float32")
    fixed_embedding = fixed_embedding / np.linalg.norm(fixed_embedding)

    bbox = (300, 200, 400, 300)
    for _ in range(100):
        det = make_detection(bbox, matched_person_id=None, match_score=0.0,
                              embedding=fixed_embedding)
        process_frame(conn, tracker, presence, recognizer, [det], now, notify_events)
        now += timedelta(milliseconds=500)

    unknown_clusters = db.list_persons(conn, is_known=False)
    assert len(unknown_clusters) == 1, "no deben crearse clusters nuevos para la misma cara"
    assert len(notify_events) == 1
    assert notify_events[0]["is_known"] is False
    assert notify_events[0]["person_id"] == unknown_clusters[0]["id"]
