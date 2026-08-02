from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app.config import settings
from app.quality import counters as quality_counters
from app.state import state
from app.templating import templates

router = APIRouter()

EDITABLE_FIELDS = [
    "RTSP_URL", "TARGET_FPS", "MIN_FACE_PX", "DET_SCORE_MIN", "BLUR_VAR_MIN",
    "SIM_THRESHOLD", "MARGIN_THRESHOLD", "UNKNOWN_CLUSTER_THRESHOLD",
    "UNKNOWN_TTL_HOURS", "UNKNOWN_RETENTION_DAYS",
    "MIN_FRAMES_CONFIRM", "IOU_THRESHOLD", "TRACK_MAX_AGE",
    "PRESENCE_TIMEOUT_MIN", "NOTIFY_COOLDOWN_MIN", "BATCH_WINDOW_SEC",
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
]

FIELD_TYPES = {
    "TARGET_FPS": float, "MIN_FACE_PX": int, "DET_SCORE_MIN": float, "BLUR_VAR_MIN": float,
    "SIM_THRESHOLD": float, "MARGIN_THRESHOLD": float, "UNKNOWN_CLUSTER_THRESHOLD": float,
    "UNKNOWN_TTL_HOURS": float, "UNKNOWN_RETENTION_DAYS": float,
    "MIN_FRAMES_CONFIRM": int, "IOU_THRESHOLD": float, "TRACK_MAX_AGE": int,
    "PRESENCE_TIMEOUT_MIN": float, "NOTIFY_COOLDOWN_MIN": float, "BATCH_WINDOW_SEC": float,
}


def _rewrite_env_file(values: dict[str, str]) -> None:
    env_path = Path(".env")
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    keys_written = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
        key = stripped.split("=", 1)[0]
        if key in values:
            new_lines.append(f"{key}={values[key]}")
            keys_written.add(key)
        else:
            new_lines.append(line)
    for key, value in values.items():
        if key not in keys_written:
            new_lines.append(f"{key}={value}")
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


@router.get("/settings", response_class=HTMLResponse)
async def settings_view(request: Request):
    current = {field: getattr(settings, field) for field in EDITABLE_FIELDS}
    pipeline_stats = state.worker.stats.as_dict() if state.worker is not None else {}
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "active": "settings", "current": current,
            "pipeline_stats": pipeline_stats, "quality_stats": quality_counters.as_dict(),
        },
    )


@router.post("/settings")
async def update_settings(request: Request):
    form = await request.form()
    to_write = {}
    for field in EDITABLE_FIELDS:
        if field not in form:
            continue
        raw = form[field]
        caster = FIELD_TYPES.get(field, str)
        try:
            value = caster(raw) if raw != "" else getattr(settings, field)
        except ValueError:
            continue
        setattr(settings, field, value)
        to_write[field] = raw
    _rewrite_env_file(to_write)
    return RedirectResponse("/settings", status_code=303)


@router.post("/settings/test-telegram")
async def test_telegram():
    if state.notifier is None:
        return JSONResponse({"ok": False, "message": "notifier no disponible"}, status_code=503)
    ok, message = await state.notifier.test_connection()
    return JSONResponse({"ok": ok, "message": message})
