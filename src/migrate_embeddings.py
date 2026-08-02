"""Recalcula todos los embeddings guardados en la base (known_faces y
pending_faces) con el motor de reconocimiento actual. Hace falta correrlo
UNA VEZ despues de cambiar el modelo de embeddings (ej. de SFace a
EdgeFace), porque los vectores viejos quedan en un espacio distinto y ya no
son comparables con los nuevos -- dejarlos sin migrar puede producir
matches incorrectos silenciosos.

Recalcula a partir de las imagenes ya guardadas en disco (crop_path), no
hace falta volver a pasar a nadie por la camara. Si a algun registro le
falta el archivo o no se le vuelve a detectar un rostro adentro, se
descarta (mejor eso que dejar un embedding inservible dando falsos
matches) y se avisa para que lo reenroles a mano.

Los pending_faces que estaban agrupados en clusters quedan liberados
(cluster_id a NULL) porque los grupos se armaron con el modelo viejo. Hay
que volver a correr cluster_pending.py despues de esto.

Uso: python src/migrate_embeddings.py
"""
import os
import cv2

from config import load_config
from db import Database
from face_engine import FaceEngine


def reembed(engine: FaceEngine, image_path: str):
    if not image_path or not os.path.exists(image_path):
        return None
    frame = cv2.imread(image_path)
    if frame is None:
        return None
    faces = engine.detect(frame)
    if len(faces) == 0:
        return None
    face_row = max(faces, key=lambda f: f[-1])
    return engine.embed(frame, face_row)


def main():
    cfg = load_config()
    db = Database(cfg["paths"]["db_path"])
    engine = FaceEngine(cfg)

    print("[migrate] recalculando embeddings de known_faces...")
    known = db.get_all_known_faces_full()
    ok, failed = 0, []
    for known_id, name, crop_path in known:
        emb = reembed(engine, crop_path)
        if emb is None:
            failed.append((known_id, name, crop_path))
            continue
        db.update_known_embedding(known_id, emb)
        ok += 1
    print(f"[migrate] known_faces: {ok} recalculados, {len(failed)} descartados")
    if failed:
        db.delete_known_faces([f[0] for f in failed])
        print("[migrate] hay que reenrolar a mano (no se pudo recalcular el embedding):")
        for _known_id, name, crop_path in failed:
            print(f"  [!] {name} -- imagen: {crop_path!r}")

    print("[migrate] recalculando embeddings de pending_faces...")
    pending = db.get_all_pending_with_crops()
    ok, failed_ids = 0, []
    for pending_id, crop_path in pending:
        emb = reembed(engine, crop_path)
        if emb is None:
            failed_ids.append(pending_id)
            continue
        db.update_pending_embedding(pending_id, emb)
        ok += 1
    print(f"[migrate] pending_faces: {ok} recalculados, {len(failed_ids)} descartados")
    if failed_ids:
        db.delete_pending_faces(failed_ids)

    n_freed = db.reset_pending_clusters()
    print(
        f"[migrate] {n_freed} rostros pendientes liberados de sus clusters viejos. "
        f"Corre cluster_pending.py de nuevo cuando quieras revisar la webUI."
    )


if __name__ == "__main__":
    main()
