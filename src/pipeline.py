"""Loop principal: toma frames de la camara, detecta rostros, los compara
contra la base de conocidos y notifica por Telegram. Los rostros que no
matchean con nadie se guardan (con dedupe) en el pool de 'pendientes' para
clasificar despues con cluster_pending.py + la app de revision."""
import os
import time
import uuid
import cv2

from config import load_config
from db import Database
from face_engine import FaceEngine
from capture import RTSPStream
from telegram_notify import TelegramNotifier


def is_far_from_recent(embedding, recent_embeddings, min_distance):
    """True si el embedding es 'distinto' a todos los recientes (distancia
    coseno = 1 - similitud). Sirve para no guardar 40 fotos casi iguales
    del mismo desconocido parado frente a camara."""
    for other in recent_embeddings:
        sim = FaceEngine.cosine_similarity(embedding, other)
        distance = 1.0 - sim
        if distance < min_distance:
            return False
    return True


def main():
    cfg = load_config()
    db = Database(cfg["paths"]["db_path"])
    engine = FaceEngine(cfg)
    notifier = TelegramNotifier(
        cfg["notify"]["telegram_bot_token"],
        cfg["notify"]["telegram_chat_id"],
        send_photo=cfg["notify"]["send_photo"],
    )

    stream = RTSPStream(cfg["rtsp"]["url"], cfg["rtsp"]["reconnect_delay_sec"]).start()
    print("[pipeline] esperando primer frame de la camara...")
    while stream.read() is None:
        time.sleep(0.5)
    print("[pipeline] stream activo, arrancando deteccion")

    last_notified = {}  # nombre_conocido -> timestamp ultimo aviso
    # Para desconocidos no alcanza con una clave fija: serian todos "la misma
    # persona" para el cooldown. Guardamos (embedding, timestamp) de cada
    # desconocido notificado recientemente y comparamos por similitud.
    recent_unknown_notified = []  # [(embedding, timestamp), ...]

    frame_interval = cfg["processing"]["frame_interval_sec"]
    known_threshold = cfg["matching"]["known_threshold"]
    dedupe_min_distance = cfg["pending"]["dedupe_min_distance"]
    recent_buffer_size = cfg["pending"]["recent_buffer_size"]
    # Umbral de distancia para decidir si un desconocido "ya fue notificado
    # hace poco" o es alguien distinto. Reusa el mismo criterio del dedupe
    # del pool de pendientes, se puede separar en config.yaml si hace falta.
    unknown_notify_distance = cfg["notify"].get("unknown_notify_distance", 0.40)

    def cooldown_ok(key, cooldown_sec):
        now = time.time()
        last = last_notified.get(key, 0)
        if now - last >= cooldown_sec:
            last_notified[key] = now
            return True
        return False

    def unknown_cooldown_ok(embedding, cooldown_sec):
        """True si este rostro desconocido NO fue notificado hace menos de
        cooldown_sec (comparando por similitud de embedding, no por una
        clave global)."""
        nonlocal recent_unknown_notified
        now = time.time()
        # Purga entradas vencidas
        recent_unknown_notified = [
            (emb, ts) for emb, ts in recent_unknown_notified if now - ts < cooldown_sec
        ]
        for emb, _ts in recent_unknown_notified:
            sim = FaceEngine.cosine_similarity(embedding, emb)
            if (1.0 - sim) < unknown_notify_distance:
                return False  # ya se notifico a alguien muy parecido hace poco
        recent_unknown_notified.append((embedding, now))
        return True

    while True:
        loop_start = time.time()
        frame = stream.read()
        if frame is None:
            time.sleep(0.2)
            continue

        known_faces = db.get_all_known_faces()
        recent_pending = db.get_recent_pending_embeddings(recent_buffer_size)

        detections = engine.detect(frame)
        for face_row in detections:
            embedding = engine.embed(frame, face_row)
            name, sim = engine.match_known(embedding, known_faces, known_threshold)

            if name is not None:
                # Persona conocida
                if cooldown_ok(name, cfg["notify"]["known_cooldown_sec"]):
                    crop = engine.crop_face(frame, face_row)
                    snap_path = os.path.join(
                        cfg["paths"]["snapshots_dir"], f"{name}_{int(time.time())}.jpg"
                    )
                    cv2.imwrite(snap_path, crop)
                    db.add_event(name, sim, snap_path)
                    hora = time.strftime("%Y-%m-%d %H:%M:%S")
                    caption = f"Persona detectada: {name}\nHora: {hora}\nConfianza: {sim:.2f}"
                    notifier.send_photo(snap_path, caption)
                    print(f"[pipeline] {hora} - conocido: {name} ({sim:.2f})")
            else:
                # Desconocido: revisar si vale la pena guardarlo (dedupe)
                if is_far_from_recent(embedding, recent_pending, dedupe_min_distance):
                    crop = engine.crop_face(frame, face_row)
                    crop_name = f"{uuid.uuid4().hex}.jpg"
                    crop_path = os.path.join(cfg["paths"]["pending_crops_dir"], crop_name)
                    cv2.imwrite(crop_path, crop)
                    db.add_pending_face(embedding, crop_path)
                    recent_pending.append(embedding)  # para no comparar contra si mismo en este mismo frame

                if unknown_cooldown_ok(embedding, cfg["notify"]["unknown_cooldown_sec"]):
                    crop = engine.crop_face(frame, face_row)
                    snap_path = os.path.join(
                        cfg["paths"]["snapshots_dir"], f"desconocido_{int(time.time())}.jpg"
                    )
                    cv2.imwrite(snap_path, crop)
                    db.add_event("desconocido", sim, snap_path)
                    hora = time.strftime("%Y-%m-%d %H:%M:%S")
                    caption = f"Persona desconocida detectada\nHora: {hora}"
                    notifier.send_photo(snap_path, caption)
                    print(f"[pipeline] {hora} - desconocido (mejor sim: {sim:.2f})")

        elapsed = time.time() - loop_start
        time.sleep(max(0.0, frame_interval - elapsed))


if __name__ == "__main__":
    main()
