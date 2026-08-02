import sqlite3
from dataclasses import dataclass

from app.capture import CaptureWorker
from app.notifier import Notifier


@dataclass
class AppState:
    conn: sqlite3.Connection | None = None
    worker: CaptureWorker | None = None
    notifier: Notifier | None = None


state = AppState()
