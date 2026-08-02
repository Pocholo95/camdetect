import logging

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger("face_presence.quality")


class QualityCounters:
    """Contadores de descartes por motivo, para /settings estadísticas."""

    def __init__(self):
        self.total = 0
        self.passed = 0
        self.rejected_det_score = 0
        self.rejected_size = 0
        self.rejected_blur = 0
        self.rejected_pose = 0
        self.rejected_edge = 0

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "rejected_det_score": self.rejected_det_score,
            "rejected_size": self.rejected_size,
            "rejected_blur": self.rejected_blur,
            "rejected_pose": self.rejected_pose,
            "rejected_edge": self.rejected_edge,
        }


counters = QualityCounters()


def _blur_variance(gray_crop: np.ndarray) -> float:
    if gray_crop.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


def _pose_ratio(kps: np.ndarray) -> float:
    """Ratio de distancia horizontal ojo-nariz izquierda vs derecha.

    kps: 5 landmarks InsightFace, orden [left_eye, right_eye, nose, left_mouth, right_mouth].
    """
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    d_left = abs(nose[0] - left_eye[0])
    d_right = abs(nose[0] - right_eye[0])
    lo, hi = sorted([max(d_left, 1e-6), max(d_right, 1e-6)])
    return hi / lo


def evaluate_face(frame: np.ndarray, bbox: np.ndarray, det_score: float,
                   kps: np.ndarray) -> tuple[bool, float, str | None]:
    """Aplica el filtro de calidad obligatorio (spec 4.1).

    Devuelve (accepted, quality_scalar, reject_reason).
    """
    counters.total += 1
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    h, w = frame.shape[:2]
    x1c, y1c, x2c, y2c = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    width, height = x2c - x1c, y2c - y1c

    if det_score < settings.DET_SCORE_MIN:
        counters.rejected_det_score += 1
        logger.debug("descartada: det_score %.3f < %.3f", det_score, settings.DET_SCORE_MIN)
        return False, 0.0, "det_score"

    if width < settings.MIN_FACE_PX or height < settings.MIN_FACE_PX:
        counters.rejected_size += 1
        logger.debug("descartada: tamano %dx%d < %d px", width, height, settings.MIN_FACE_PX)
        return False, 0.0, "size"

    edge_margin = 2
    if x1 <= edge_margin or y1 <= edge_margin or x2 >= w - edge_margin or y2 >= h - edge_margin:
        counters.rejected_edge += 1
        logger.debug("descartada: bbox toca el borde del frame")
        return False, 0.0, "edge"

    if width <= 0 or height <= 0:
        counters.rejected_size += 1
        return False, 0.0, "size"

    crop = frame[y1c:y2c, x1c:x2c]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_var = _blur_variance(gray)
    if blur_var < settings.BLUR_VAR_MIN:
        counters.rejected_blur += 1
        logger.debug("descartada: blur_var %.2f < %.2f", blur_var, settings.BLUR_VAR_MIN)
        return False, 0.0, "blur"

    ratio = _pose_ratio(kps)
    if ratio > 3.0:
        counters.rejected_pose += 1
        logger.debug("descartada: pose extrema ratio %.2f > 3.0", ratio)
        return False, 0.0, "pose"

    quality = _quality_scalar(det_score, width, height, blur_var)
    counters.passed += 1
    return True, quality, None


def _quality_scalar(det_score: float, width: int, height: int, blur_var: float) -> float:
    size_score = min(1.0, ((width + height) / 2) / 200.0)
    sharpness_score = min(1.0, blur_var / 200.0)
    quality = 0.5 * det_score + 0.3 * size_score + 0.2 * sharpness_score
    return float(max(0.0, min(1.0, quality)))
