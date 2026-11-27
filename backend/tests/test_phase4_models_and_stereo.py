from __future__ import annotations

import json
import math

import numpy as np
from fastapi.testclient import TestClient

from app.ai.autolearn import (
    GOLDEN_DATASET_DIRNAME,
    _load_golden_set_excluded_ids,
    enforce_pseudo_label_ratio,
    should_promote_checkpoint,
)
from app.main import app
from app.vision.classification import (
    MACRO_CLOUD_CLASSES_5,
    classify_cloud_type,
    map_to_macro_cloud_class,
)
from app.vision.color import analyze_color, normalize_ae_awb_bgr
from app.vision.geometry import CameraGeometry, triangulate_cloud_base_altitude_m, undistort_sky_frame
from app.vision.segmentation import NOT_SKY_IGNORE_INDEX, reduce_segmentation_to_3_classes


def test_reduced_3class_segmentation_and_5class_macro_classification():
    # 5-class internal mask: 0=clear, 1=thin, 2=dense, 3=dark, 4=glare
    mask5 = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)
    valid = np.array([[255, 255, 255], [255, 255, 0]], dtype=np.uint8)
    m3, pcts = reduce_segmentation_to_3_classes(mask5, valid)
    assert m3[0, 0] == 0
    assert m3[0, 1] == 1
    assert m3[0, 2] == 2 and m3[1, 0] == 2
    assert m3[1, 1] == NOT_SKY_IGNORE_INDEX  # sun glare excluded
    assert m3[1, 2] == NOT_SKY_IGNORE_INDEX  # masked obstacle excluded
    assert pcts == {"clear_sky_pct": 25.0, "cloud_pct": 25.0, "thick_cloud_pct": 50.0}

    assert map_to_macro_cloud_class("cirrus") == "high_cirriform"
    assert map_to_macro_cloud_class("cumulus") == "cumuliform_low"
    assert map_to_macro_cloud_class("stratus") == "stratiform_low"
    assert map_to_macro_cloud_class("altocumulus") == "mid_level"
    assert map_to_macro_cloud_class("cumulonimbus_like") == "convective_storm"
    assert map_to_macro_cloud_class("unknown") == "unknown"

    img = np.full((64, 64, 3), (210, 150, 90), dtype=np.uint8)
    cov = {"cloud_coverage_pct": 2.0, "clear_sky_pct": 98.0, "thin_cloud_pct": 2.0, "dense_cloud_pct": 0.0, "dark_cloud_pct": 0.0}
    col = analyze_color(img, np.full((64, 64), 255, dtype=np.uint8))
    tex = {"edge_density": 0.01, "local_contrast": 3.0, "fragmentation_index": 0.05}
    cls = classify_cloud_type(img, cov, col, tex)
    assert cls["macro_cloud_class"] in MACRO_CLOUD_CLASSES_5 or cls["macro_cloud_class"] == "unknown"


def test_ae_awb_normalization_and_lens_undistortion():
    # Two images with artificial warm/dark AE+AWB shift must converge closer after normalization
    base = np.full((128, 128, 3), (200, 200, 200), dtype=np.uint8)  # Neutral bright overcast cloud
    shifted = np.clip(base.astype(np.float32) * np.array([0.82, 0.90, 1.05]), 0, 255).astype(np.uint8)
    norm_img, meta = normalize_ae_awb_bgr(shifted)
    assert norm_img.shape == shifted.shape
    assert meta["b_gain"] > meta["r_gain"]  # Compensates warm tint
    assert abs(float(norm_img[:, :, 0].mean()) - float(norm_img[:, :, 2].mean())) < 15.0

    geom = CameraGeometry(
        elevation_deg=60.0,
        azimuth_deg=0.0,
        roll_deg=0.0,
        fov_horizontal_deg=80.0,
        fov_vertical_deg=55.0,
        cloud_altitude_m=2000.0,
        tilt_correction_enabled=True,
    )
    # Zero coefficients -> exact no-op
    same = undistort_sky_frame(base, geom)
    assert same is base
    # Non-zero radial distortion runs cleanly
    undist = undistort_sky_frame(base, geom, k1=-0.18, k2=0.04)
    assert undist.shape == base.shape


def test_stereo_cloud_base_triangulation_analytical_and_api(isolated_env):
    # Place cloud target at exactly East=0 m, North=1500 m, Altitude=2000 m above cam1 (lat=45.0, lon=10.0)
    # Place cam2 1000 m East of cam1
    cam1_lat, cam1_lon = 45.0, 10.0
    cos_lat = math.cos(math.radians(cam1_lat))
    cam2_lat, cam2_lon = 45.0, 10.0 + (1000.0 / (111320.0 * cos_lat))

    # Ray 1 from (0,0,0) to (0,1500,2000): az=0 deg, el=atan2(2000, 1500)
    el1 = math.degrees(math.atan2(2000.0, 1500.0))
    az1 = 0.0
    # Ray 2 from (1000,0,0) to (0,1500,2000): dx=-1000, dy=1500, horiz=hypot(1000,1500)
    el2 = math.degrees(math.atan2(2000.0, math.hypot(-1000.0, 1500.0)))
    az2 = (math.degrees(math.atan2(-1000.0, 1500.0)) + 360.0) % 360.0

    res = triangulate_cloud_base_altitude_m(cam1_lat, cam1_lon, el1, az1, cam2_lat, cam2_lon, el2, az2)
    assert res["valid"] is True
    assert abs(res["altitude_m"] - 2000.0) < 10.0
    assert abs(res["baseline_m"] - 1000.0) < 5.0
    assert res["ray_miss_m"] < 5.0

    with TestClient(app) as client:
        r = client.post(
            "/api/camera/stereo/triangulate",
            json={
                "cam1_lat": cam1_lat,
                "cam1_lon": cam1_lon,
                "el1_deg": el1,
                "az1_deg": az1,
                "cam2_lat": cam2_lat,
                "cam2_lon": cam2_lon,
                "el2_deg": el2,
                "az2_deg": az2,
            },
        )
        assert r.status_code == 200
        assert abs(r.json()["altitude_m"] - 2000.0) < 10.0


def test_autotrain_golden_exclusion_50pct_pseudo_cap_and_category_gate(isolated_env):
    # 1) Golden set exclusion loader
    g_dir = isolated_env.datasets_dir / GOLDEN_DATASET_DIRNAME
    g_dir.mkdir(parents=True, exist_ok=True)
    (g_dir / "manifest.json").write_text(
        json.dumps({"items": [{"observation_id": 77, "source_raw_path": "/tmp/golden_77.jpg"}]}),
        encoding="utf-8",
    )
    g_ids, g_names = _load_golden_set_excluded_ids(isolated_env)
    assert 77 in g_ids and "golden_77.jpg" in g_names

    # 2) <= 50% pseudo-label cap in training split
    items = (
        [{"split": "train", "user_labeled": True, "group_key": "u1"} for _ in range(4)]
        + [{"split": "train", "user_labeled": False, "group_key": "loc"} for _ in range(20)]
        + [{"split": "val", "user_labeled": False, "group_key": "holdout"} for _ in range(5)]
    )
    capped = enforce_pseudo_label_ratio(items, max_pseudo_fraction=0.50)
    tr_items = [x for x in capped if x["split"] == "train"]
    pseudo_tr = [x for x in tr_items if not x["user_labeled"]]
    assert len(pseudo_tr) / len(tr_items) <= 0.50 + 1e-9
    assert len([x for x in capped if x["split"] == "val"]) == 5  # Val untouched

    # 3) Per-category regression promotion gate blocks checkpoint that hurts twilight despite higher average
    ok, reason = should_promote_checkpoint(
        candidate_metric=0.88,
        active_metric=0.85,
        candidate_by_category={"clear": 0.95, "twilight": 0.62},
        active_by_category={"clear": 0.90, "twilight": 0.78},
        max_category_regression=0.03,
    )
    assert ok is False and "twilight" in reason

    ok2, _ = should_promote_checkpoint(
        candidate_metric=0.88,
        active_metric=0.85,
        candidate_by_category={"clear": 0.93, "twilight": 0.77},
        active_by_category={"clear": 0.90, "twilight": 0.78},
        max_category_regression=0.03,
    )
    assert ok2 is True
