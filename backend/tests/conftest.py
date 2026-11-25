from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import reset_config_manager
from app.database import init_database


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    cfg_path = tmp_path / "config.yaml"
    monkeypatch.setenv("SKYOBSERVER_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SKYOBSERVER_DB_PATH", str(data_dir / "db" / "test.db"))
    monkeypatch.setenv("SKYOBSERVER_CONFIG_PATH", str(cfg_path))
    monkeypatch.setenv("SKYOBSERVER_DISABLE_WORKER", "1")
    monkeypatch.setenv("SKYOBSERVER_OFFLINE_MODE", "true")

    mgr = reset_config_manager(cfg_path)
    cfg = mgr.get()
    cfg.ensure_directories()
    init_database(cfg.db_path)
    yield cfg
