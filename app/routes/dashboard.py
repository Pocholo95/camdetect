from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from app import db
from app.state import state
from app.templating import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    conn = state.conn
    present = []
    if state.worker is not None:
        for person_id, last_seen, n_frames in state.worker.presence.currently_present():
            person = db.get_person(conn, person_id)
            if person is not None:
                present.append({
                    "name": person["name"],
                    "is_known": bool(person["is_known"]),
                    "last_seen": last_seen.strftime("%H:%M:%S"),
                    "n_frames": n_frames,
                })

    recent_logs = db.list_presence_logs(conn, limit=20)
    recent = []
    for log in recent_logs:
        person = db.get_person(conn, log["person_id"])
        recent.append({
            "name": person["name"] if person else "?",
            "is_known": bool(person["is_known"]) if person else False,
            "first_seen": log["first_seen"],
            "last_seen": log["last_seen"],
            "n_frames": log["n_frames"],
            "best_score": log["best_score"],
        })

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {"active": "dashboard", "present": present, "recent": recent},
    )


def _mjpeg_generator():
    import time
    while True:
        if state.worker is not None:
            frame = state.worker.latest_jpeg()
            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
        time.sleep(0.4)


@router.get("/stream.mjpg")
async def stream():
    return StreamingResponse(
        _mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )
