"""Wrapper sobre los modelos nativos de OpenCV: YuNet (deteccion) y SFace
(embeddings). Ambos corren en CPU sin dependencias externas mas alla de
opencv-contrib-python.
"""
import cv2
import numpy as np


class FaceEngine:
    def __init__(self, cfg: dict):
        det_cfg = cfg["detection"]
        self.detector = cv2.FaceDetectorYN.create(
            det_cfg["yunet_model"],
            "",
            (det_cfg["input_width"], det_cfg["input_height"]),
            score_threshold=det_cfg["score_threshold"],
            nms_threshold=det_cfg["nms_threshold"],
            top_k=5000,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(det_cfg["sface_model"], "")
        self._size = (det_cfg["input_width"], det_cfg["input_height"])

    def detect(self, frame: np.ndarray):
        """Devuelve la lista de detecciones crudas de YuNet para este frame.
        Cada fila: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt,
        x_rcm, y_rcm, x_lcm, y_lcm, score]
        """
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []
        return faces

    def embed(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        """Alinea y recorta el rostro, y devuelve su embedding (vector)."""
        aligned = self.recognizer.alignCrop(frame, face_row)
        feature = self.recognizer.feature(aligned)
        return feature.flatten()

    def crop_face(
        self,
        frame: np.ndarray,
        face_row: np.ndarray,
        margin: float = 0.3,
        min_size: int = 240,
    ) -> np.ndarray:
        """Recorte simple (sin alinear) con margen, para guardar como
        snapshot legible por un humano. Si el recorte queda mas chico que
        min_size (camaras/streams de baja resolucion), lo agranda con
        interpolacion cubica: no agrega detalle real, pero evita mandar a
        Telegram o mostrar en la webUI una miniatura minuscula."""
        x, y, w, h = face_row[:4].astype(int)
        mx, my = int(w * margin), int(h * margin)
        H, W = frame.shape[:2]
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(W, x + w + mx), min(H, y + h + my)
        crop = frame[y0:y1, x0:x1].copy()

        crop_h, crop_w = crop.shape[:2]
        largest_side = max(crop_h, crop_w)
        if largest_side > 0 and largest_side < min_size:
            scale = min(min_size / largest_side, 4.0)  # tope para no inventar detalle
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return crop

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = a.flatten().astype(np.float64)
        b = b.flatten().astype(np.float64)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def match_known(self, embedding: np.ndarray, known_faces, threshold: float):
        """known_faces: lista de (id, name, embedding). Devuelve
        (name, similarity) del mejor match si supera el umbral, si no
        (None, mejor_similarity_encontrada)."""
        best_name, best_sim = None, -1.0
        for _id, name, emb in known_faces:
            sim = self.cosine_similarity(embedding, emb)
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_sim >= threshold:
            return best_name, best_sim
        return None, best_sim
