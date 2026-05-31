"""SQLAlchemy engine and session factory."""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from ecoforo.config import config

engine = create_engine(
    config.DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in config.DATABASE_URL else {},
)

# Enable WAL mode and foreign keys for SQLite
if "sqlite" in config.DATABASE_URL:

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()


SessionLocal = sessionmaker(bind=engine)


def get_db() -> Session:
    """Yield a database session, closing after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
