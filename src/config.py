"""Carga la configuracion desde config.yaml y expone rutas absolutas."""
import os
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_config(path: str = None) -> dict:
    path = path or os.path.join(_ROOT, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolver rutas relativas contra la raiz del proyecto
    def abspath(rel):
        return rel if os.path.isabs(rel) else os.path.join(_ROOT, rel)

    cfg["detection"]["yunet_model"] = abspath(cfg["detection"]["yunet_model"])
    cfg["detection"]["sface_model"] = abspath(cfg["detection"]["sface_model"])
    cfg["paths"]["data_dir"] = abspath(cfg["paths"]["data_dir"])
    cfg["paths"]["db_path"] = abspath(cfg["paths"]["db_path"])
    cfg["paths"]["pending_crops_dir"] = abspath(cfg["paths"]["pending_crops_dir"])
    cfg["paths"]["known_crops_dir"] = abspath(cfg["paths"]["known_crops_dir"])
    cfg["paths"]["snapshots_dir"] = abspath(cfg["paths"]["snapshots_dir"])

    for d in (
        cfg["paths"]["data_dir"],
        cfg["paths"]["pending_crops_dir"],
        cfg["paths"]["known_crops_dir"],
        cfg["paths"]["snapshots_dir"],
    ):
        os.makedirs(d, exist_ok=True)

    return cfg


PROJECT_ROOT = _ROOT
