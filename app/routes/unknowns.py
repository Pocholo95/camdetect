from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app import db, unknowns
from app.state import state
from app.templating import templates

router = APIRouter()


@router.get("/unknowns", response_class=HTMLResponse)
async def list_unknowns(request: Request):
    conn = state.conn
    clusters = []
    for row in db.unreviewed_clusters(conn):
        sightings = db.cluster_embeddings(conn, row["cluster_id"])
        best = max(sightings, key=lambda s: s["quality"]) if sightings else None
        clusters.append({
            "cluster_id": row["cluster_id"],
            "name": row["name"],
            "n_sightings": row["n_sightings"],
            "last_seen": row["last_seen"],
            "face_crop": best["face_crop"] if best else None,
            "full_frame": best["full_frame"] if best else None,
        })
    known_people = db.list_persons(conn, is_known=True)
    return templates.TemplateResponse(
        request,
        "unknowns.html",
        {"active": "unknowns", "clusters": clusters, "known_people": known_people},
    )


def _merge_cluster_into_person(conn, cluster_id: int, target_person_id: int) -> None:
    sightings = db.cluster_embeddings(conn, cluster_id)
    for s in sightings:
        vector = db.blob_to_vector(s["vector"])
        db.add_embedding(conn, target_person_id, vector, condition=None, source="auto_label")
    db.mark_reviewed(conn, [s["id"] for s in sightings])
    db.delete_person(conn, cluster_id)


# Las rutas literales /unknowns/batch/* deben registrarse antes que
# /unknowns/{cluster_id}/*: si no, "batch" se intenta parsear como cluster_id
# y FastAPI responde 422 en vez de llegar a estos endpoints.
#
# Los IDs seleccionados se leen a mano de la form data (en vez de usar
# `list[int] = Form(...)`) para no depender de como FastAPI/python-multipart
# resuelvan claves repetidas segun la version instalada.

async def _selected_ids(request: Request) -> list[int]:
    form = await request.form()
    return [int(v) for v in form.getlist("selected") if str(v).strip() != ""]


@router.post("/unknowns/batch/assign")
async def batch_assign(request: Request):
    conn = state.conn
    form = await request.form()
    selected = [int(v) for v in form.getlist("selected") if str(v).strip() != ""]
    target_person_id_raw = form.get("target_person_id")
    if not selected or not target_person_id_raw:
        return RedirectResponse("/unknowns", status_code=303)
    target_person_id = int(target_person_id_raw)
    for cluster_id in selected:
        _merge_cluster_into_person(conn, cluster_id, target_person_id)
    if state.worker is not None:
        state.worker.recognizer.invalidate_known_cache()
    return RedirectResponse("/unknowns", status_code=303)


@router.post("/unknowns/batch/create")
async def batch_create(request: Request):
    conn = state.conn
    form = await request.form()
    selected = [int(v) for v in form.getlist("selected") if str(v).strip() != ""]
    name = (form.get("name") or "").strip()
    if not selected or not name:
        return RedirectResponse("/unknowns", status_code=303)
    new_person_id = db.create_person(conn, name=name, is_known=True, notify=True)
    for cluster_id in selected:
        _merge_cluster_into_person(conn, cluster_id, new_person_id)
    if state.worker is not None:
        state.worker.recognizer.invalidate_known_cache()
    return RedirectResponse("/unknowns", status_code=303)


@router.post("/unknowns/batch/ignore")
async def batch_ignore(request: Request):
    conn = state.conn
    for cluster_id in await _selected_ids(request):
        sightings = db.cluster_embeddings(conn, cluster_id)
        db.mark_reviewed(conn, [s["id"] for s in sightings])
    return RedirectResponse("/unknowns", status_code=303)


@router.post("/unknowns/batch/delete")
async def batch_delete(request: Request):
    conn = state.conn
    for cluster_id in await _selected_ids(request):
        unknowns.delete_cluster(conn, cluster_id)
    return RedirectResponse("/unknowns", status_code=303)


@router.post("/unknowns/{cluster_id}/assign")
async def assign_to_existing(cluster_id: int, target_person_id: int = Form(...)):
    conn = state.conn
    _merge_cluster_into_person(conn, cluster_id, target_person_id)
    if state.worker is not None:
        state.worker.recognizer.invalidate_known_cache()
    return RedirectResponse("/unknowns", status_code=303)


@router.post("/unknowns/{cluster_id}/create")
async def create_from_cluster(cluster_id: int, name: str = Form(...)):
    conn = state.conn
    sightings = db.cluster_embeddings(conn, cluster_id)
    db.promote_unknown_to_named(conn, cluster_id, name)
    for s in sightings:
        vector = db.blob_to_vector(s["vector"])
        db.add_embedding(conn, cluster_id, vector, condition=None, source="auto_label")
    db.mark_reviewed(conn, [s["id"] for s in sightings])
    if state.worker is not None:
        state.worker.recognizer.invalidate_known_cache()
    return RedirectResponse("/unknowns", status_code=303)


@router.post("/unknowns/{cluster_id}/ignore")
async def ignore_cluster(cluster_id: int):
    conn = state.conn
    sightings = db.cluster_embeddings(conn, cluster_id)
    db.mark_reviewed(conn, [s["id"] for s in sightings])
    return RedirectResponse("/unknowns", status_code=303)


@router.post("/unknowns/{cluster_id}/delete")
async def delete_cluster_route(cluster_id: int):
    unknowns.delete_cluster(state.conn, cluster_id)
    return RedirectResponse("/unknowns", status_code=303)
