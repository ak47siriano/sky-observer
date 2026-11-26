"""SQLite database engine, WAL configuration, and session helpers."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_config_manager
from app.models_db import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _configure_sqlite_pragma(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.execute("PRAGMA busy_timeout=5000;")
    cursor.close()


def _ensure_schema_migrations(engine: Engine) -> None:
    """Apply safe, idempotent SQLite ALTER TABLE migrations on existing databases."""
    with engine.begin() as conn:
        cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(events)").fetchall()}
        if cols:
            if "user_status" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE events ADD COLUMN user_status VARCHAR(24) NOT NULL DEFAULT 'pending'"
                )
                conn.exec_driver_sql("UPDATE events SET user_status = 'confirmed' WHERE acknowledged = 1")
            if "user_note" not in cols:
                conn.exec_driver_sql("ALTER TABLE events ADD COLUMN user_note TEXT NOT NULL DEFAULT ''")

        fc_cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(forecasts)").fetchall()}
        if fc_cols and "baseline_metrics_json" not in fc_cols:
            conn.exec_driver_sql("ALTER TABLE forecasts ADD COLUMN baseline_metrics_json TEXT NOT NULL DEFAULT '{}'")

        fv_cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(forecast_verifications)").fetchall()}
        if fv_cols:
            if "persistence_abs_error" not in fv_cols:
                conn.exec_driver_sql("ALTER TABLE forecast_verifications ADD COLUMN persistence_abs_error FLOAT")
            if "optical_flow_abs_error" not in fv_cols:
                conn.exec_driver_sql("ALTER TABLE forecast_verifications ADD COLUMN optical_flow_abs_error FLOAT")


def init_database(db_path: Path | None = None) -> Engine:
    global _engine, _SessionLocal
    cfg = get_config_manager().get()
    target_path = (db_path or cfg.db_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{target_path.as_posix()}"

    if _engine is not None and str(_engine.url) == url:
        Base.metadata.create_all(bind=_engine)
        _ensure_schema_migrations(_engine)
        return _engine

    if _engine is not None:
        _engine.dispose()

    _engine = create_engine(
        url,
        connect_args={"check_same_thread": False},
        future=True,
    )
    event.listen(_engine, "connect", _configure_sqlite_pragma)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False)
    Base.metadata.create_all(bind=_engine)
    _ensure_schema_migrations(_engine)
    return _engine


def get_engine() -> Engine:
    if _engine is None:
        return init_database()
    return _engine


@contextmanager
def get_db_session() -> Iterator[Session]:
    if _SessionLocal is None:
        init_database()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a database session."""
    if _SessionLocal is None:
        init_database()
    assert _SessionLocal is not None
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()
