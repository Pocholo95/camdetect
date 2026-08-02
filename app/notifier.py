import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from app import db
from app.config import settings

logger = logging.getLogger("face_presence.notifier")

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


@dataclass
class NotificationEvent:
    person_id: int
    name: str
    is_known: bool
    score: float
    snapshot_path: str | None
    timestamp: str  # ISO


class Notifier:
    """Consumidor de eventos de notificacion: cooldown, batching y envio a Telegram."""

    def __init__(self, queue_dir: str | None = None):
        self._last_notified: dict[int, datetime] = {}
        self._pending: list[NotificationEvent] = []
        self._batch_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._client: httpx.AsyncClient | None = None
        queue_dir = queue_dir or settings.MEDIA_DIR
        self._disk_queue_path = Path(queue_dir) / "pending_notifications.jsonl"
        self._disk_queue_path.parent.mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        self._client = httpx.AsyncClient(timeout=15.0)
        await self._flush_disk_queue()

    async def stop(self, conn=None) -> None:
        if self._batch_task is not None:
            self._batch_task.cancel()
        if self._pending:
            events, self._pending = self._pending, []
            try:
                await self._send_batch(conn, events)
            except Exception:
                logger.warning("no se pudieron enviar notificaciones pendientes al cerrar")
        if self._client is not None:
            await self._client.aclose()

    def in_cooldown(self, person_id: int, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        last = self._last_notified.get(person_id)
        if last is None:
            return False
        return (now - last) < timedelta(minutes=settings.NOTIFY_COOLDOWN_MIN)

    async def notify(self, conn, event: NotificationEvent) -> None:
        """Encola un evento de track confirmado. Respeta cooldown por identidad."""
        now = datetime.now(timezone.utc)
        if self.in_cooldown(event.person_id, now):
            logger.debug("suprimido por cooldown: %s", event.name)
            return

        async with self._lock:
            already_queued = any(e.person_id == event.person_id for e in self._pending)
            if already_queued:
                return
            self._pending.append(event)
            self._last_notified[event.person_id] = now
            if self._batch_task is None:
                self._batch_task = asyncio.create_task(self._flush_after_delay(conn))

    async def _flush_after_delay(self, conn) -> None:
        try:
            await asyncio.sleep(settings.BATCH_WINDOW_SEC)
        except asyncio.CancelledError:
            return
        async with self._lock:
            events = self._pending
            self._pending = []
            self._batch_task = None
        if events:
            await self._send_batch(conn, events)

    def _format_message(self, conn, events: list[NotificationEvent]) -> str:
        now_local = datetime.now().strftime("%H:%M")
        lines = [f"🏠 {now_local} — {len(events)} persona(s) detectada(s)", ""]
        for e in events:
            if e.is_known:
                lines.append(f"✅ {e.name} (confianza {e.score:.2f})")
            else:
                seen_today = db.count_sightings_today(conn, e.person_id) if conn else "?"
                lines.append(f"❓ {e.name} (visto {seen_today} veces hoy)")
        return "\n".join(lines)

    async def _send_batch(self, conn, events: list[NotificationEvent]) -> None:
        text = self._format_message(conn, events)
        photos = [e.snapshot_path for e in events if e.snapshot_path]

        try:
            if len(photos) >= 2:
                await self._send_media_group(photos, text)
            elif len(photos) == 1:
                await self._send_photo(photos[0], text)
            else:
                await self._send_message(text)
            logger.info("notificacion enviada: %d persona(s)", len(events))
        except Exception as exc:
            logger.warning("fallo enviando a Telegram, encolando en disco: %s", exc)
            self._enqueue_disk(events, text)

    async def _request_with_retry(self, method: str, files=None, data=None,
                                   max_retries: int = 5) -> httpx.Response:
        url = TELEGRAM_API.format(token=settings.TELEGRAM_BOT_TOKEN, method=method)
        delay = 1.0
        for attempt in range(max_retries):
            resp = await self._client.post(url, data=data, files=files)
            if resp.status_code == 429:
                retry_after = resp.json().get("parameters", {}).get("retry_after", delay)
                logger.warning("Telegram 429, esperando %.1fs", retry_after)
                await asyncio.sleep(retry_after)
                continue
            if resp.status_code >= 500:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError(f"Telegram request failed after {max_retries} retries: {method}")

    async def _send_message(self, text: str) -> None:
        await self._request_with_retry(
            "sendMessage", data={"chat_id": settings.TELEGRAM_CHAT_ID, "text": text}
        )

    async def _send_photo(self, photo_path: str, caption: str) -> None:
        with open(photo_path, "rb") as f:
            await self._request_with_retry(
                "sendPhoto",
                data={"chat_id": settings.TELEGRAM_CHAT_ID, "caption": caption},
                files={"photo": f},
            )

    async def _send_media_group(self, photo_paths: list[str], caption: str) -> None:
        media = []
        files = {}
        for i, path in enumerate(photo_paths):
            key = f"photo{i}"
            media.append({
                "type": "photo",
                "media": f"attach://{key}",
                **({"caption": caption, "parse_mode": "HTML"} if i == 0 else {}),
            })
            files[key] = open(path, "rb")
        try:
            await self._request_with_retry(
                "sendMediaGroup",
                data={"chat_id": settings.TELEGRAM_CHAT_ID, "media": json.dumps(media)},
                files=files,
            )
        finally:
            for f in files.values():
                f.close()

    def _enqueue_disk(self, events: list[NotificationEvent], text: str) -> None:
        with open(self._disk_queue_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"text": text, "events": [asdict(e) for e in events]}) + "\n")

    async def _flush_disk_queue(self) -> None:
        if not self._disk_queue_path.exists():
            return
        lines = self._disk_queue_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return
        remaining = []
        for line in lines:
            try:
                item = json.loads(line)
                photos = [
                    e.get("snapshot_path") for e in item["events"] if e.get("snapshot_path")
                ]
                if len(photos) >= 2:
                    await self._send_media_group(photos, item["text"])
                elif len(photos) == 1:
                    await self._send_photo(photos[0], item["text"])
                else:
                    await self._send_message(item["text"])
            except Exception as exc:
                logger.warning("no se pudo reenviar notificacion en cola de disco: %s", exc)
                remaining.append(line)
        if remaining:
            self._disk_queue_path.write_text("\n".join(remaining) + "\n", encoding="utf-8")
        else:
            self._disk_queue_path.unlink(missing_ok=True)

    async def test_connection(self) -> tuple[bool, str]:
        try:
            resp = await self._client.get(
                TELEGRAM_API.format(token=settings.TELEGRAM_BOT_TOKEN, method="getMe")
            )
            resp.raise_for_status()
            await self._send_message("✅ face-presence: prueba de conexion Telegram OK")
            return True, "Conexion exitosa"
        except Exception as exc:
            return False, str(exc)
