import numpy as np

from app.config import settings
from app.quality import evaluate_face


def _kps_frontal(cx, cy):
    return np.array([
        [cx - 10, cy - 5], [cx + 10, cy - 5], [cx, cy],
        [cx - 8, cy + 10], [cx + 8, cy + 10],
    ], dtype="float32")


def test_small_face_is_rejected_before_matcher():
    """Spec 12.4: una cara de 20 px se descarta antes de llegar al matcher."""
    frame = np.random.default_rng(1).integers(0, 255, (480, 640, 3), dtype="uint8")
    size = 20
    assert size < settings.MIN_FACE_PX
    x1, y1 = 300, 300
    bbox = np.array([x1, y1, x1 + size, y1 + size], dtype="float32")
    kps = _kps_frontal(x1 + size / 2, y1 + size / 2)

    accepted, quality, reason = evaluate_face(frame, bbox, det_score=0.95, kps=kps)

    assert accepted is False
    assert reason == "size"
    assert quality == 0.0


def test_low_det_score_is_rejected():
    frame = np.random.default_rng(2).integers(0, 255, (480, 640, 3), dtype="uint8")
    bbox = np.array([200, 200, 280, 280], dtype="float32")
    kps = _kps_frontal(240, 240)

    accepted, _, reason = evaluate_face(frame, bbox, det_score=0.3, kps=kps)

    assert accepted is False
    assert reason == "det_score"


def test_face_touching_edge_is_rejected():
    frame = np.random.default_rng(3).integers(0, 255, (480, 640, 3), dtype="uint8")
    bbox = np.array([0, 100, 80, 180], dtype="float32")
    kps = _kps_frontal(40, 140)

    accepted, _, reason = evaluate_face(frame, bbox, det_score=0.9, kps=kps)

    assert accepted is False
    assert reason == "edge"
