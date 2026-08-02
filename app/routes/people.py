import cv2
import numpy as np
from fastapi import APIRouter, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app import db
from app.capture import save_crop
from app.config import settings
from app.quality import evaluate_face
from app.recognizer import l2_normalize
from app.state import state
from app.templating import templates

router = APIRouter()


@router.get("/people", response_class=HTMLResponse)
async def list_people(request: Request):
    conn = state.conn
    persons = []
    for p in db.list_persons(conn, is_known=True):
        embeddings = db.list_embeddings(conn, p["id"])
        persons.append({"row": p, "n_embeddings": len(embeddings)})
    return templates.TemplateResponse(
        request, "people.html", {"active": "people", "persons": persons}
    )


@router.post("/people/new")
async def create_person(name: str = Form(...)):
    conn = state.conn
    if db.get_person_by_name(conn, name) is not None:
        return RedirectResponse("/people", status_code=303)
    person_id = db.create_person(conn, name=name, is_known=True, notify=True)
    return RedirectResponse(f"/people/{person_id}", status_code=303)


@router.get("/people/{person_id}", response_class=HTMLResponse)
async def person_detail(request: Request, person_id: int):
    conn = state.conn
    person = db.get_person(conn, person_id)
    embeddings = db.list_embeddings(conn, person_id)
    return templates.TemplateResponse(
        request,
        "person_detail.html",
        {"active": "people", "person": person, "embeddings": embeddings},
    )


@router.post("/people/{person_id}/notify")
async def toggle_notify(person_id: int, notify: bool = Form(False)):
    db.set_person_notify(state.conn, person_id, notify)
    return RedirectResponse(f"/people/{person_id}", status_code=303)


@router.post("/people/{person_id}/delete")
async def delete_person_route(person_id: int):
    db.delete_person(state.conn, person_id)
    if state.worker is not None:
        state.worker.recognizer.invalidate_known_cache()
    return RedirectResponse("/people", status_code=303)


@router.post("/people/{person_id}/capture")
async def capture_from_stream(person_id: int, condition: str = Form("day")):
    """Toma el frame actual del stream y guarda el embedding de la mejor cara."""
    if state.worker is None:
        return JSONResponse({"ok": False, "message": "worker no disponible"}, status_code=503)

    frame_jpeg = state.worker.latest_jpeg()
    if frame_jpeg is None:
        return JSONResponse({"ok": False, "message": "no hay frame disponible aun"})

    frame = cv2.imdecode(np.frombuffer(frame_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    return _save_best_face(person_id, frame, condition, source="capture")


@router.post("/people/{person_id}/upload")
async def upload_photo(person_id: int, condition: str = Form("day"), file: UploadFile = None):
    if file is None:
        return JSONResponse({"ok": False, "message": "sin archivo"}, status_code=400)
    content = await file.read()
    frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"ok": False, "message": "imagen invalida"}, status_code=400)
    return _save_best_face(person_id, frame, condition, source="upload")


def _save_best_face(person_id: int, frame: np.ndarray, condition: str, source: str):
    conn = state.conn
    recognizer = state.worker.recognizer if state.worker is not None else None
    if recognizer is None:
        return JSONResponse({"ok": False, "message": "reconocedor no disponible"}, status_code=503)

    faces = recognizer.get_faces(frame)
    if not faces:
        return JSONResponse({"ok": False, "message": "no se detecto ninguna cara"})

    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    accepted, quality, reason = evaluate_face(
        frame, best_face.bbox, best_face.det_score, best_face.kps
    )
    if not accepted:
        return JSONResponse({
            "ok": False,
            "quality": quality,
            "message": f"cara descartada por filtro de calidad ({reason})",
        })

    embedding = l2_normalize(best_face.embedding)
    db.add_embedding(conn, person_id, embedding, condition, source)

    x1, y1, x2, y2 = [int(v) for v in best_face.bbox[:4]]
    h, w = frame.shape[:2]
    crop = frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
    save_crop(crop, settings.MEDIA_DIR, f"person{person_id}")

    if state.worker is not None:
        state.worker.recognizer.invalidate_known_cache()

    return JSONResponse({"ok": True, "quality": quality})


@router.post("/embeddings/{embedding_id}/delete")
async def delete_embedding_route(embedding_id: int, person_id: int = Form(...)):
    db.delete_embedding(state.conn, embedding_id)
    if state.worker is not None:
        state.worker.recognizer.invalidate_known_cache()
    return RedirectResponse(f"/people/{person_id}", status_code=303)
