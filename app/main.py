import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import db, unknowns
from app.capture import CaptureWorker
from app.config import settings
from app.notifier import Notifier
from app.routes import dashboard, logs, people, settings_routes
from app.routes import unknowns as unknowns_routes
from app.state import state

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("face_presence.main")

CLEANUP_INTERVAL_SEC = 24 * 3600


async def _daily_cleanup_loop(conn) -> None:
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL_SEC)
            unknowns.cleanup_stale_clusters(conn, settings.MEDIA_DIR)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("error en limpieza diaria de desconocidos")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.MEDIA_DIR).mkdir(parents=True, exist_ok=True)

    conn = db.get_connection()
    db.init_db(conn)
    state.conn = conn

    notifier = Notifier()
    await notifier.start()
    state.notifier = notifier

    loop = asyncio.get_event_loop()
    worker = CaptureWorker(loop, notifier)
    worker.start()
    state.worker = worker

    cleanup_task = asyncio.create_task(_daily_cleanup_loop(conn))

    logger.info("face-presence iniciado en puerto %d", settings.WEB_PORT)

    yield

    logger.info("apagando face-presence (SIGTERM/shutdown)...")
    cleanup_task.cancel()
    worker.stop()
    await notifier.stop(conn)
    conn.close()


Path(settings.MEDIA_DIR).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Face Presence", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/media", StaticFiles(directory=settings.MEDIA_DIR), name="media")

app.include_router(dashboard.router)
app.include_router(people.router)
app.include_router(unknowns_routes.router)
app.include_router(logs.router)
app.include_router(settings_routes.router)
