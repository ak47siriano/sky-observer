"""CLI helper to prepare, remap, and register public datasets (SWINySEG, Clouds-1500, Almería ASI, TJNU MGCD/GCD)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ai.datasets import import_public_dataset
from app.config import get_config_manager
from app.database import get_db_session, init_database


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare & import public sky/cloud datasets into Sky Observer.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["swinyseg", "clouds1000", "clouds1500", "almeria_asi", "tjnu_mgcd", "tjnu_gcd"],
        help="Public dataset format to remap and import.",
    )
    parser.add_argument("--path", required=True, help="Path to local extracted dataset folder.")
    parser.add_argument("--name", default=None, help="Optional custom dataset name in DB.")
    args = parser.parse_args()

    cfg = get_config_manager().get()
    cfg.ensure_directories()
    init_database(cfg.db_path)

    with get_db_session() as db:
        rec = import_public_dataset(db=db, config=cfg, dataset_kind=args.dataset, source_dir=Path(args.path), name=args.name)
        print(f"Imported dataset '{rec.name}' (id={rec.id}, items={rec.item_count}, tasks={rec.task_types_json})")


if __name__ == "__main__":
    main()
