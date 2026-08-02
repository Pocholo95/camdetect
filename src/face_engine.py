"""Wrapper sobre YuNet (deteccion, via cv2.FaceDetectorYN) y EdgeFace
(embeddings, via ONNX Runtime). Ambos corren en CPU.

EdgeFace no tiene wrapper nativo en OpenCV como SFace (cv2.FaceRecognizerSF),
asi que la alineacion de 5 puntos y el preprocesamiento se hacen a mano:
- Alineacion: transformacion de similitud (Umeyama) de los 5 landmarks de
  YuNet contra la plantilla ArcFace de referencia en 112x112, mismo orden
  (ojo derecho, ojo izquierdo, nariz, comisura derecha, comisura izquierda)
  que ya usa YuNet.
- Preprocesamiento: BGR->RGB, (pixel - 127.5) / 127.5, NCHW.
- Embedding de salida: 512-D, normalizado L2 antes de comparar.
"""
import cv2
import numpy as np
import onnxruntime as ort

# Plantilla ArcFace de referencia en 112x112. El orden de los 5 puntos tiene
# que coincidir con el orden en que YuNet devuelve sus landmarks (ver
# detect()): ojo derecho, ojo izquierdo, nariz, comisura derecha, comisura
# izquierda. Es la misma plantilla estandar que usan insightface/ArcFace.
REFERENCE_5PT = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estima la transformacion de similitud (rotacion+escala+traslacion)
    que mejor mapea src -> dst (algoritmo de Umeyama). Devuelve una matriz
    2x3 lista para cv2.warpAffine."""
    n, dim = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(cov)
    d = np.ones(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[-1] = -1
    R = U @ np.diag(d) @ Vt
    var_src = (src_c ** 2).sum() / n
    scale = (S * d).sum() / var_src if var_src > 0 else 1.0
    t = dst_mean - scale * R @ src_mean
    M = np.zeros((2, 3), dtype=np.float64)
    M[:2, :2] = scale * R
    M[:, 2] = t
    return M


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
        # intra_op_num_threads explicito: si no se fija, onnxruntime intenta
        # anclar sus hilos a nucleos de CPU especificos (sched_setaffinity),
        # lo que falla con "Invalid argument" dentro de contenedores LXC/
        # Docker sin privilegios donde el set de CPUs visible no se puede
        # afinar. Fijarlo a mano evita ese codepath (y de paso limita cuantos
        # hilos usa, importante en un LXC con pocos nucleos).
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = int(det_cfg.get("onnx_threads", 2))
        self.session = ort.InferenceSession(
            det_cfg["edgeface_model"], sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self.session.get_inputs()[0].name
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

    def align_face(self, frame: np.ndarray, face_row: np.ndarray, image_size: int = 112) -> np.ndarray:
        """Alinea el rostro a 112x112 usando los 5 landmarks de YuNet contra
        la plantilla ArcFace de referencia."""
        landmarks = face_row[4:14].reshape(5, 2).astype(np.float64)
        M = _similarity_transform(landmarks, REFERENCE_5PT)
        return cv2.warpAffine(frame, M, (image_size, image_size), borderValue=0.0)

    def embed(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        """Alinea el rostro y devuelve su embedding (vector 512-D, normalizado L2)."""
        aligned = self.align_face(frame, face_row)
        blob = cv2.dnn.blobFromImage(
            aligned,
            scalefactor=1.0 / 127.5,
            size=(112, 112),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        out = self.session.run(None, {self._input_name: blob})[0]
        vec = out.flatten().astype(np.float64)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

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
