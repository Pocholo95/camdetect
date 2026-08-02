from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app import db
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


@router.post("/unknowns/{cluster_id}/assign")
async def assign_to_existing(cluster_id: int, target_person_id: int = Form(...)):
    conn = state.conn
    sightings = db.cluster_embeddings(conn, cluster_id)
    for s in sightings:
        vector = db.blob_to_vector(s["vector"])
        db.add_embedding(conn, target_person_id, vector, condition=None, source="auto_label")
    db.mark_reviewed(conn, [s["id"] for s in sightings])
    db.delete_person(conn, cluster_id)
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
