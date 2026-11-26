"""Bootstrap local sky dataset and build the protected default baseline PyTorch models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.ai.datasets import bootstrap_local_sky_dataset  # noqa: E402
from app.ai.device import get_hardware_status  # noqa: E402
from app.ai.registry import BASELINE_DEFAULT_MARKER, get_model_cache  # noqa: E402
from app.ai.trainer import get_training_manager  # noqa: E402
from app.config import get_config_manager  # noqa: E402
from app.database import get_db_session, init_database  # noqa: E402
from app.models_db import ModelRegistry  # noqa: E402


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
    # Keep built-in radiometric-physics-v1 active for Stage-1 segmentation by registering
    # baseline segmentation in standby, while activating classification, temporal & event defaults.
    jobs = [
        ("segmentation", 4, "CloudSegUNet-Mobile", False),
        ("classification", 6, "CloudTypeClassifier-Fusion", True),
        ("temporal", 12, "TemporalSkyConvGRU", True),
        ("event", 12, "SkyEventClassifier-GRU", True),
    ]
    for family, epochs, arch_override, activate_default in jobs:
        print(f"Training default baseline {family} ({arch_override}, {epochs} epochs)...")
        res = tm.start_training(
            model_family=family,
            dataset_id=ds_id,
            epochs=epochs,
            batch_size=4,
            learning_rate=0.002,
            auto_activate=activate_default,
            async_run=False,
            architecture_override=arch_override,
        )
        with get_db_session() as db:
            rec = (
                db.query(ModelRegistry)
                .filter(ModelRegistry.training_session_id == res["session_id"])
                .order_by(ModelRegistry.id.desc())
                .first()
            )
            if rec is not None:
                if BASELINE_DEFAULT_MARKER not in (rec.notes or ""):
                    rec.notes = f"{BASELINE_DEFAULT_MARKER} Default baseline model ({rec.notes})"
                db.commit()
        print("Finished job:", res)

    with get_db_session() as db:
        get_model_cache().refresh_from_db(db)
        rows = db.query(ModelRegistry).all()
        for r in rows:
            print(f"Registered: {r.model_id} | arch={r.architecture} | active={r.is_active} | metrics={r.metrics_json}")


if __name__ == "__main__":
    main()
