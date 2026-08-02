from collections import defaultdict

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from app import db
from app.state import state
from app.templating import templates

router = APIRouter()


@router.get("/logs", response_class=HTMLResponse)
async def logs_view(
    request: Request,
    person_id: int | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
):
    conn = state.conn
    rows = db.list_presence_logs(
        conn, person_id=person_id, date_from=date_from, date_to=date_to, limit=500
    )

    logs = []
    by_day: dict[str, int] = defaultdict(int)
    for r in rows:
        person = db.get_person(conn, r["person_id"])
        logs.append({
            "name": person["name"] if person else "?",
            "is_known": bool(person["is_known"]) if person else False,
            "first_seen": r["first_seen"],
            "last_seen": r["last_seen"],
            "n_frames": r["n_frames"],
            "best_score": r["best_score"],
            "snapshot": r["snapshot"],
        })
        by_day[r["first_seen"][:10]] += 1

    timeline = sorted(by_day.items(), reverse=True)
    persons = db.list_persons(conn)

    return templates.TemplateResponse(
        request,
        "logs.html",
        {
            "active": "logs", "logs": logs, "timeline": timeline,
            "persons": persons, "filter_person_id": person_id,
            "filter_date_from": date_from or "", "filter_date_to": date_to or "",
        },
    )
