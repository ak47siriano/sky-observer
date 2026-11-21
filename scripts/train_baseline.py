"""Bootstrap local sky dataset and run initial fine-tuning for all 4 local PyTorch models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ai.datasets import bootstrap_local_sky_dataset
from app.ai.device import get_hardware_status
from app.ai.trainer import get_training_manager
from app.config import get_config_manager
from app.database import get_db_session, init_database
from app.models_db import ModelRegistry


def main() -> None:
    cfg = get_config_manager().get()
    cfg.ensure_directories()
    init_database(cfg.db_path)

    hw = get_hardware_status(cfg.ai.device, cfg.ai.max_gpu_memory_fraction, cfg.vision.inference_resolution)
    print("Hardware:", json.dumps(hw, indent=2))

    with get_db_session() as db:
        ds = bootstrap_local_sky_dataset(db, cfg, sequences_per_scenario=3)
        ds_id = ds.id
        print(f"Bootstrapped dataset id={ds_id} name={ds.name} items={ds.item_count}")

    tm = get_training_manager()
    for family, epochs in [("segmentation", 4), ("classification", 6), ("temporal", 12), ("event", 12)]:
        print(f"Training {family} ({epochs} epochs)...")
        res = tm.start_training(
            model_family=family,
            dataset_id=ds_id,
            epochs=epochs,
            batch_size=4,
            learning_rate=0.002,
            auto_activate=True,
            async_run=False,
        )
        print("Finished job:", res)

    with get_db_session() as db:
        rows = db.query(ModelRegistry).all()
        for r in rows:
            print(f"Registered: {r.model_id} | active={r.is_active} | metrics={r.metrics_json}")


if __name__ == "__main__":
    main()
