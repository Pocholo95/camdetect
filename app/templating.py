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


templates.env.filters["media_url"] = media_url
