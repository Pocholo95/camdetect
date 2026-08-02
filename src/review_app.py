"""Mini web local para revisar los clusters de rostros desconocidos que
armo cluster_pending.py, ponerles nombre (con lo cual pasan a la base de
conocidos) o descartarlos.

Uso: python src/review_app.py
Despues entrar a http://IP_DEL_LXC:5000
"""
import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

from config import load_config, PROJECT_ROOT
from db import Database

cfg = load_config()
db = Database(cfg["paths"]["db_path"])

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
)


@app.route("/")
def index():
    pending_clusters = db.get_clusters(status="pending")
    clusters_data = []
    for cid, status, label, created_at in pending_clusters:
        faces = db.get_cluster_faces(cid)
        clusters_data.append(
            {
                "id": cid,
                "n_faces": len(faces),
                "thumbnails": [os.path.basename(f[2]) for f in faces[:6]],
            }
        )
    return render_template("review.html", clusters=clusters_data)


@app.route("/crop/<path:filename>")
def serve_crop(filename):
    return send_from_directory(cfg["paths"]["pending_crops_dir"], filename)


@app.route("/cluster/<int:cluster_id>")
def view_cluster(cluster_id):
    faces = db.get_cluster_faces(cluster_id)
    thumbnails = [os.path.basename(f[2]) for f in faces]
    return render_template("cluster.html", cluster_id=cluster_id, thumbnails=thumbnails)


@app.route("/cluster/<int:cluster_id>/label", methods=["POST"])
def label_cluster(cluster_id):
    name = request.form.get("name", "").strip()
    if not name:
        return redirect(url_for("view_cluster", cluster_id=cluster_id))

    faces = db.get_cluster_faces(cluster_id)  # [(pending_id, embedding, crop_path), ...]

    # Se guarda un known_face por cada embedding del cluster (mas robusto
    # para el matching que promediar, ya que conserva la variedad de
    # angulos/luz vista).
    for _pending_id, embedding, crop_path in faces:
        db.add_known_face(name, embedding, crop_path=crop_path)

    db.label_cluster(cluster_id, name)
    db.delete_pending_faces([f[0] for f in faces])
    return redirect(url_for("index"))


@app.route("/cluster/<int:cluster_id>/discard", methods=["POST"])
def discard_cluster(cluster_id):
    faces = db.get_cluster_faces(cluster_id)
    db.discard_cluster(cluster_id)
    db.delete_pending_faces([f[0] for f in faces])
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(
        host=cfg["flask"]["host"],
        port=cfg["flask"]["port"],
        debug=cfg["flask"]["debug"],
    )
