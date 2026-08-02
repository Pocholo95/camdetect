"""Enroll manual opcional: si ya tenes fotos de alguien y no queres
esperar a que el sistema lo capture solo, podes armar:

    known_faces_src/
        Juan/
            foto1.jpg
            foto2.jpg
        Maria/
            foto1.jpg

y correr: python src/enroll.py known_faces_src/

Cada foto debe tener un solo rostro razonablemente visible.
"""
import os
import sys
import cv2

from config import load_config
from db import Database
from face_engine import FaceEngine


def main():
    if len(sys.argv) < 2:
        print("Uso: python src/enroll.py <carpeta_con_subcarpetas_por_persona>")
        sys.exit(1)

    src_dir = sys.argv[1]
    cfg = load_config()
    db = Database(cfg["paths"]["db_path"])
    engine = FaceEngine(cfg)

    total = 0
    for person_name in sorted(os.listdir(src_dir)):
        person_dir = os.path.join(src_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        for fname in sorted(os.listdir(person_dir)):
            fpath = os.path.join(person_dir, fname)
            frame = cv2.imread(fpath)
            if frame is None:
                print(f"  [!] no se pudo leer {fpath}, se salta")
                continue

            faces = engine.detect(frame)
            if len(faces) == 0:
                print(f"  [!] no se detecto rostro en {fpath}, se salta")
                continue
            if len(faces) > 1:
                print(f"  [!] mas de un rostro en {fpath}, se usa el de mayor score")

            face_row = max(faces, key=lambda f: f[-1])
            embedding = engine.embed(frame, face_row)
            db.add_known_face(person_name, embedding, crop_path=fpath)
            total += 1
            print(f"  + {person_name}: {fname}")

    print(f"[enroll] listo, {total} embeddings agregados")


if __name__ == "__main__":
    main()
