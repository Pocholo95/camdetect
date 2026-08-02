import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app import db
from app.config import settings


@pytest.fixture
def conn(tmp_path):
    db_path = tmp_path / "test.db"
    old_db_path = settings.DB_PATH
    settings.DB_PATH = str(db_path)

    connection = db.get_connection()
    db.init_db(connection)
    yield connection

    connection.close()
    settings.DB_PATH = old_db_path


@pytest.fixture
def restore_settings():
    """Permite a un test mutar `settings` y garantiza que se revierte al final."""
    from app.config import Settings

    snapshot = Settings().model_dump()
    yield settings
    for key, value in snapshot.items():
        setattr(settings, key, value)
