"""Agrupa los rostros 'pendientes' (desconocidos guardados por el
pipeline) usando DBSCAN sobre distancia coseno entre embeddings. Cada
cluster deberia corresponder, en teoria, a una misma persona vista varias
veces. Correr manualmente o via cron/systemd timer.

Uso: python src/cluster_pending.py
"""
import numpy as np
from sklearn.cluster import DBSCAN

from config import load_config
from db import Database


def main():
    cfg = load_config()
    db = Database(cfg["paths"]["db_path"])

    pending = db.get_unclustered_pending()
    if not pending:
        print("[cluster] no hay rostros pendientes sin clusterizar")
        return

    ids = [p[0] for p in pending]
    embeddings = np.stack([p[1] for p in pending]).astype(np.float64)

    eps = cfg["clustering"]["eps"]
    min_samples = cfg["clustering"]["min_samples"]

    # metric='cosine' calcula directamente 1 - similitud_coseno (misma
    # escala que pending.dedupe_min_distance y notify.unknown_notify_distance
    # en el resto del sistema), sin necesidad de normalizar a mano.
    # eps=0.35 agrupa rostros con similitud coseno aproximada >= 0.65.
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embeddings)
    labels = clustering.labels_  # -1 = ruido (no forma cluster con nadie)

    label_to_cluster_id = {}
    n_new_clusters = 0
    n_noise = 0

    for pending_id, label in zip(ids, labels):
        if label == -1:
            n_noise += 1
            continue  # se deja sin cluster_id, queda disponible para la proxima corrida
        if label not in label_to_cluster_id:
            label_to_cluster_id[label] = db.create_cluster()
            n_new_clusters += 1
        db.assign_cluster(pending_id, label_to_cluster_id[label])

    print(
        f"[cluster] {len(pending)} rostros pendientes -> "
        f"{n_new_clusters} clusters nuevos, {n_noise} sin agrupar (quedan pendientes)"
    )


if __name__ == "__main__":
    main()
