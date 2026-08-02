"""Crea el esquema de base de datos en DB_PATH si no existe."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db
from app.config import settings


def main() -> None:
    conn = db.get_connection()
    db.init_db(conn)
    print(f"Base de datos inicializada en {settings.DB_PATH}")
    conn.close()


if __name__ == "__main__":
    main()
