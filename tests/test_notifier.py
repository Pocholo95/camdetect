import asyncio

import pytest

from app.config import settings
from app.notifier import NotificationEvent, Notifier


@pytest.mark.asyncio
async def test_cooldown_suppresses_repeat_notification(conn, tmp_path):
    notifier = Notifier(queue_dir=str(tmp_path))
    sent = []
    notifier._send_message = lambda text: sent.append(text) or asyncio.sleep(0)

    old_batch_window = settings.BATCH_WINDOW_SEC
    settings.BATCH_WINDOW_SEC = 0.05
    try:
        event = NotificationEvent(
            person_id=1, name="Juan", is_known=True, score=0.8,
            snapshot_path=None, timestamp="2026-01-01T00:00:00",
        )
        await notifier.notify(conn, event)
        await asyncio.sleep(0.1)
        assert len(sent) == 1

        # Segunda aparicion inmediata de la misma identidad: debe suprimirse por cooldown.
        await notifier.notify(conn, event)
        await asyncio.sleep(0.1)
        assert len(sent) == 1, "el cooldown debe impedir una segunda notificacion inmediata"
    finally:
        settings.BATCH_WINDOW_SEC = old_batch_window


@pytest.mark.asyncio
async def test_batch_window_groups_multiple_identities_in_one_send(conn, tmp_path):
    notifier = Notifier(queue_dir=str(tmp_path))
    batches = []

    async def fake_send_batch(conn, events):
        batches.append(list(events))

    notifier._send_batch = fake_send_batch

    old_batch_window = settings.BATCH_WINDOW_SEC
    settings.BATCH_WINDOW_SEC = 0.1
    try:
        e1 = NotificationEvent(1, "Juan", True, 0.8, None, "2026-01-01T00:00:00")
        e2 = NotificationEvent(2, "Ana", True, 0.7, None, "2026-01-01T00:00:01")
        await notifier.notify(conn, e1)
        await notifier.notify(conn, e2)
        await asyncio.sleep(0.2)

        assert len(batches) == 1
        assert len(batches[0]) == 2
    finally:
        settings.BATCH_WINDOW_SEC = old_batch_window
