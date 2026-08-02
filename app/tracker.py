from dataclasses import dataclass, field

import numpy as np

UNKNOWN_PERSON_ID = -1  # marcador interno; no se persiste tal cual


@dataclass
class Detection:
    bbox: tuple
    quality: float
    face_crop: np.ndarray
    full_frame: np.ndarray
    embedding: np.ndarray
    matched_person_id: int | None  # None = desconocido para esta cara
    match_score: float


@dataclass
class Track:
    track_id: int
    bbox: tuple
    last_seen_frame: int
    votes: dict = field(default_factory=dict)          # person_id -> suma de scores
    unknown_votes: float = 0.0
    n_frames: int = 0
    best_quality: float = 0.0
    best_face_crop: np.ndarray | None = None
    best_full_frame: np.ndarray | None = None
    embeddings: list = field(default_factory=list)
    notified: bool = False
    log_id: int | None = None
    person_id: int | None = None                       # persons.id si es cluster de desconocido asignado en runtime
    confirmed_identity: int | None = None               # identidad resuelta (argmax) del ultimo update
    unknown_cluster_id: int | None = None                # cluster asignado la primera vez que se detecto desconocido

    def resolved_known_identity(self) -> int | None:
        """Argmax de votes (solo personas conocidas), si supera unknown_votes.

        None => sin evidencia suficiente de una persona conocida; el track se
        trata como desconocido (ver unknown_cluster_id en capture.py).
        """
        if not self.votes:
            return None
        best_person_id = max(self.votes, key=self.votes.get)
        if self.votes[best_person_id] > self.unknown_votes:
            return best_person_id
        return None

    def best_identity_score(self) -> float:
        n = max(self.n_frames, 1)
        if self.confirmed_identity is not None and self.confirmed_identity in self.votes:
            return self.votes[self.confirmed_identity] / n
        return 0.0


def iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


class Tracker:
    """Tracker ligero por IoU + votacion de identidad (spec seccion 5)."""

    def __init__(self, iou_threshold: float, max_age: int):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self.frame_idx = 0

    def _new_track_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def _associate(self, detections: list[Detection]) -> dict:
        """Devuelve {detection_index: track_id} para las mejores parejas por IoU."""
        assignment = {}
        used_tracks = set()
        pairs = []
        for di, det in enumerate(detections):
            for tid, track in self.tracks.items():
                score = iou(det.bbox, track.bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, di, tid))
        pairs.sort(key=lambda p: p[0], reverse=True)
        used_dets = set()
        for score, di, tid in pairs:
            if di in used_dets or tid in used_tracks:
                continue
            assignment[di] = tid
            used_dets.add(di)
            used_tracks.add(tid)
        return assignment

    def step(self, detections: list[Detection]) -> tuple[list[Track], list[Track]]:
        """Procesa un frame de detecciones ya filtradas por calidad.

        Devuelve (tracks_actualizados, tracks_muertos_este_frame).
        """
        self.frame_idx += 1
        assignment = self._associate(detections)

        updated_tracks: list[Track] = []

        for di, det in enumerate(detections):
            tid = assignment.get(di)
            if tid is None:
                tid = self._new_track_id()
                self.tracks[tid] = Track(
                    track_id=tid, bbox=det.bbox, last_seen_frame=self.frame_idx
                )
            track = self.tracks[tid]
            track.bbox = det.bbox
            track.last_seen_frame = self.frame_idx
            track.n_frames += 1
            track.embeddings.append(det.embedding)

            if det.matched_person_id is not None:
                track.votes[det.matched_person_id] = (
                    track.votes.get(det.matched_person_id, 0.0) + det.match_score
                )
            else:
                track.unknown_votes += det.match_score if det.match_score > 0 else 0.5

            if det.quality > track.best_quality:
                track.best_quality = det.quality
                track.best_face_crop = det.face_crop
                track.best_full_frame = det.full_frame

            track.confirmed_identity = track.resolved_known_identity()
            updated_tracks.append(track)

        dead_tracks = []
        for tid in list(self.tracks.keys()):
            track = self.tracks[tid]
            if self.frame_idx - track.last_seen_frame > self.max_age:
                dead_tracks.append(track)
                del self.tracks[tid]

        return updated_tracks, dead_tracks

    def alive_tracks(self) -> list[Track]:
        return list(self.tracks.values())
