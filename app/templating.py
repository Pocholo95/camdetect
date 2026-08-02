from datetime import datetime, timezone
from pathlib import Path

from fastapi.templating import Jinja2Templates

from app.config import settings

templates = Jinja2Templates(directory="app/templates")


def media_url(path: str | None) -> str:
    if not path:
        return ""
    try:
        rel = Path(path).resolve().relative_to(Path(settings.MEDIA_DIR).resolve())
    except (ValueError, OSError):
        return ""
    return f"/media/{rel.as_posix()}"


def local_time(value, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Convierte un datetime/ISO-string en UTC (como se guarda en la DB) a la
    hora local del servidor para mostrarlo en la WebUI."""
    if not value:
        return "-"
    dt = datetime.fromisoformat(value) if isinstance(value, str) else value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime(fmt)


templates.env.filters["media_url"] = media_url
templates.env.filters["local_time"] = local_time
