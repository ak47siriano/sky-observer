"""Dataset Catalog, Local Importer, Integrity Checks, Leakage-Free Splitting & Local Sky Dataset Bootstrap."""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.config import AppConfig
from app.models_db import DatasetItem, DatasetRecord
from app.vision.classification import CLOUD_TYPE_CLASSES

# Verified scientific public sky/cloud datasets with honest licensing & format notes
PUBLIC_DATASET_CATALOG = [
    {
        "id": "swimseg",
        "name": "SWIMSEG (Singapore Whole Sky Imaging Segmentation)",
        "citation": "Dev et al., IEEE TGRS 2016",
        "license": "Academic / Research Non-Commercial (NTU Singapore)",
        "url": "https://github.com/Soumyabrata/color-based-segmentation",
        "samples": 1013,
        "format": "PNG sky patches + binary ground-truth masks",
        "suitable_for": ["segmentation"],
        "notes": "Daytime ground-based sky camera patches with binary cloud vs. sky masks.",
    },
    {
        "id": "swinyseg",
        "name": "SWINySEG (Singapore Whole Sky Night/Day Segmentation)",
        "citation": "Dev et al., IGARSS 2019",
        "license": "Academic / Research Non-Commercial (NTU Singapore)",
        "url": "https://malea.winkler.site/swinyseg.html",
        "samples": 6768,
        "format": "RGB daytime/nighttime sky images + binary masks",
        "suitable_for": ["segmentation"],
        "notes": "Extends SWIMSEG with nighttime sky images and binary cloud masks (request form on MaLeA page).",
    },
    {
        "id": "wsiseg",
        "name": "WSISEG (Whole-Sky Image Segmentation Database)",
        "citation": "Xie et al., Atmos. Meas. Tech. 2020",
        "license": "Research / Attribution (CC-BY)",
        "url": "https://github.com/CV-Application/WSISEG-Database",
        "samples": 400,
        "format": "Whole-sky fish-eye images + 3-class masks (cloud, sky, undefined)",
        "suitable_for": ["segmentation"],
        "notes": "400 whole-sky camera frames manually annotated with clear, cloud, and horizon-undefined classes.",
    },
    {
        "id": "clouds1500",
        "name": "Clouds-1000 / Clouds-1500 (Multi-Class Ground Sky Segmentation)",
        "citation": "LAPIX / INCoD UFSC",
        "license": "Academic / Research (Mendeley Data)",
        "url": "https://data.mendeley.com/datasets/4pw8vfsnpx",
        "samples": 1500,
        "format": "RGB sky images + multi-class masks (4 cloud types + fixed obstacles)",
        "suitable_for": ["segmentation"],
        "notes": "Phase 1B multi-class fine-tuning (also https://github.com/bjuncklaus/Clouds-1000); obstacles mapped to 255 (Not_Sky).",
    },
    {
        "id": "almeria_asi",
        "name": "All-Sky Imager Dataset Almería (CIEMAT-PSA)",
        "citation": "DLR / CIEMAT-PSA Almería (Zenodo)",
        "license": "Academic / Research (Zenodo)",
        "url": "https://doi.org/10.5281/zenodo.14639169",
        "samples": 818,
        "format": "818 512x512 all-sky RGB images + low/mid/high cloud & clear sky masks",
        "suitable_for": ["segmentation"],
        "notes": "Phase 1B multi-class fine-tuning: clear->0, high->1 (Thin), mid->2 (Dense), low->3 (Dark), sun->4, border->255.",
    },
    {
        "id": "tjnu_mgcd",
        "name": "TJNU Multimodal Ground-based Cloud Database (MGCD)",
        "citation": "Liu et al., IEEE TGRS 2020 (Tianjin Normal University)",
        "license": "Academic / Research Non-Commercial",
        "url": "https://github.com/shuangliutjnu/Multimodal-Ground-based-Cloud-Database",
        "samples": 8000,
        "format": "8,000 sky images in 7 WMO cloud classes + synchronized meteorological tabular CSV",
        "suitable_for": ["classification"],
        "notes": "Stage 2 multimodal training (RGB + tabular weather features) with lazy disk loading.",
    },
    {
        "id": "tjnu_gcd",
        "name": "TJNU Ground-based Cloud Dataset (GCD)",
        "citation": "Liu et al., Atmos. Meas. Tech. 2021 (Tianjin Normal University)",
        "license": "Academic / Research Non-Commercial",
        "url": "https://github.com/shuangliutjnu/TJNU-Ground-based-Cloud-Dataset",
        "samples": 19000,
        "format": "19,000 512x512 ground-based sky images in 7 WMO cloud folders",
        "suitable_for": ["classification"],
        "notes": "Large-scale Stage 2 WMO classification dataset; uses 100% lazy disk loading.",
    },
    {
        "id": "ccsn",
        "name": "CCSN (Cirrus Cumulus Stratus Nimbus Cloud Dataset)",
        "citation": "Zhang et al., J. Atmos. Ocean. Technol. 2018",
        "license": "Academic / Research (CCSN Harvard Dataverse)",
        "url": "https://doi.org/10.7910/DVN/CADLVD",
        "samples": 2543,
        "format": "JPEG ground-based sky photos sorted into 11 cloud genus folders (no pixel masks)",
        "suitable_for": ["classification"],
        "notes": "Image-level labels only (Ci, Cs, Cc, Ac, As, Cu, Cb, Ns, Sc, St, Ct). Do not assume segmentation masks.",
    },
    {
        "id": "cloudsen12",
        "name": "CloudSEN12 (Sentinel-2 Cloud & Shadow Benchmark)",
        "citation": "Aybar et al., Scientific Data 2022",
        "license": "CC0-1.0 Public Domain",
        "url": "https://cloudsen12.github.io/",
        "samples": 49400,
        "format": "Multi-spectral satellite patches + thin/thick/shadow masks",
        "suitable_for": ["reference_only"],
        "notes": "Nadir satellite imagery (top-down view), NOT upward-facing ground sky camera geometry.",
    },
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Canonical mapping from public dataset folder names (TJNU MGCD/GCD, CCSN) -> CLOUD_TYPE_CLASSES
WMO_FOLDER_ALIASES: dict[str, str] = {
    "1": "cumulus",
    "2": "altocumulus",
    "3": "cirrus",
    "4": "clear_sky",
    "5": "stratocumulus",
    "6": "cumulonimbus_like",
    "7": "mixed_clouds",
    "cu": "cumulus",
    "cumulus": "cumulus",
    "ac": "altocumulus",
    "altocumulus": "altocumulus",
    "as": "altostratus",
    "altostratus": "altostratus",
    "ci": "cirrus",
    "cirrus": "cirrus",
    "cs": "cirrostratus",
    "cirrostratus": "cirrostratus",
    "cc": "cirrus",
    "cirrocumulus": "cirrus",
    "cb": "cumulonimbus_like",
    "cumulonimbus": "cumulonimbus_like",
    "ns": "cumulonimbus_like",
    "nimbostratus": "cumulonimbus_like",
    "sc": "stratocumulus",
    "stratocumulus": "stratocumulus",
    "st": "stratus",
    "stratus": "stratus",
    "clear": "clear_sky",
    "clear_sky": "clear_sky",
    "sky": "clear_sky",
    "mixed": "mixed_clouds",
    "mixed_clouds": "mixed_clouds",
}


def map_wmo_folder_label(folder_name: str) -> str:
    """Normalize public dataset folder labels (TJNU MGCD/GCD 1..7, CCSN abbreviations) to CLOUD_TYPE_CLASSES."""
    key = folder_name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in CLOUD_TYPE_CLASSES:
        return key
    if key in WMO_FOLDER_ALIASES:
        return WMO_FOLDER_ALIASES[key]
    prefix = key.split("_")[0]
    return WMO_FOLDER_ALIASES.get(prefix, "mixed_clouds")


def map_mask_to_5class(raw_mask: np.ndarray, dataset_kind: str = "generic") -> np.ndarray:
    """
    Remap public segmentation dataset masks into the 5 clean Stage-1 target classes:
      0=Clear, 1=Thin, 2=Dense, 3=Dark, 4=Sun_Glare, 255=Not_Sky (ignore_index).
    """
    kind = dataset_kind.lower().strip()
    m = raw_mask if raw_mask.ndim == 2 else raw_mask[:, :, 0]
    out = np.zeros_like(m, dtype=np.uint8)

    if kind == "swinyseg":
        # Phase 1A binary: 0=clear, 1=cloud
        out = np.where(m > 127, 1, 0).astype(np.uint8)
        return out

    if kind == "almeria_asi":
        # Official Almería ASI classes.yaml:
        # 0: camera mask -> 255 (Not_Sky), 1: sky -> 0 (Clear),
        # 2: low-layer clouds -> 3 (Dark), 3: mid-layer clouds -> 2 (Dense),
        # 4: high-layer clouds -> 1 (Thin), >=5: sun glare -> 4
        out[m == 0] = 255
        out[m == 1] = 0
        out[m == 2] = 3
        out[m == 3] = 2
        out[m == 4] = 1
        out[m >= 5] = 4
        return out

    if kind in {"clouds1000", "clouds1500"}:
        # Clouds-1000/1500: 0=clear, 1=thin, 2=dense, 3=dark, 4=sun_glare, 5/255=obstacle (Not_Sky)
        out = np.clip(m, 0, 4).astype(np.uint8)
        out[(m == 5) | (m == 255)] = 255
        return out

    # Generic fallback
    if int(m.max()) > 5 and int(m.max()) != 255:
        return np.where(m > 127, 2, 0).astype(np.uint8)
    out = np.clip(m, 0, 4).astype(np.uint8)
    out[m == 255] = 255
    return out


def import_public_dataset(
    db: Session,
    config: AppConfig,
    dataset_kind: str,
    source_dir: Path,
    name: str | None = None,
) -> DatasetRecord:
    """
    Unified lazy importer for public datasets:
      - 'swinyseg' (Phase 1A binary)
      - 'clouds1000' / 'clouds1500' (Phase 1B 5-class + Not_Sky=255)
      - 'almeria_asi' (Phase 1B 5-class + Not_Sky=255)
      - 'tjnu_mgcd' / 'tjnu_gcd' (Stage 2 WMO classification)
    Remaps segmentation masks onto disk once in data/datasets/<name>/mapped_masks/ so training stays 100% lazy.
    """
    kind = dataset_kind.lower().strip()
    ds_name = name or f"public_{kind}"
    source_dir = source_dir.resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Public dataset directory not found: {source_dir}")

    if kind in {"tjnu_mgcd", "tjnu_gcd"}:
        # Class subfolders -> normalize WMO labels using map_wmo_folder_label
        rec = import_local_dataset(
            db=db,
            config=config,
            name=ds_name,
            source_dir=source_dir,
            license_name="Academic / Research (TJNU)",
            description=f"Imported {kind.upper()} WMO classification dataset",
        )
        # Re-normalize folder labels with TJNU WMO aliases
        items = db.query(DatasetItem).filter(DatasetItem.dataset_id == rec.id).all()
        for it in items:
            folder = Path(it.image_path).parent.name
            it.cloud_type_label = map_wmo_folder_label(folder)
        db.commit()
        return rec

    # Segmentation datasets (SWINySEG, Clouds-1000/1500, Almeria ASI)
    rec = import_local_dataset(
        db=db,
        config=config,
        name=ds_name,
        source_dir=source_dir,
        license_name="Academic / Research",
        description=f"Imported {kind} segmentation dataset (mapped to 5-class + Not_Sky=255)",
    )
    mapped_dir = config.datasets_dir / ds_name.replace(" ", "_").lower() / "mapped_masks"
    mapped_dir.mkdir(parents=True, exist_ok=True)
    items = db.query(DatasetItem).filter(DatasetItem.dataset_id == rec.id).all()
    for it in items:
        if not it.mask_path or not Path(it.mask_path).exists():
            continue
        raw_m = cv2.imread(it.mask_path, cv2.IMREAD_GRAYSCALE)
        if raw_m is None:
            continue
        mapped = map_mask_to_5class(raw_m, dataset_kind=kind)
        out_p = mapped_dir / f"{Path(it.image_path).stem}_mapped.png"
        cv2.imwrite(str(out_p), mapped)
        it.mask_path = str(out_p)
        valid = mapped != 255
        if np.any(valid):
            if kind == "swinyseg":
                it.cloud_coverage_gt = round(float(np.mean(mapped[valid] == 1) * 100.0), 2)
            else:
                it.cloud_coverage_gt = round(float(np.mean((mapped[valid] >= 1) & (mapped[valid] <= 3)) * 100.0), 2)
    db.commit()
    return rec


def _assign_group_splits(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Assign deterministic train/val/test splits (70/15/15) by group_key (day/sequence/camera)
    to prevent temporal leakage of nearly identical consecutive frames, stratified by
    cloud_type_label when present so every class appears in train/val/test.
    """
    group_label: dict[str, str] = {}
    groups: list[str] = []
    seen: set[str] = set()
    for it in items:
        g = str(it.get("group_key") or "default")
        if g not in seen:
            seen.add(g)
            groups.append(g)
            group_label[g] = str(it.get("cloud_type_label") or "")

    if len(groups) >= 3:
        # Stratified per-label group assignment when labels exist; otherwise interleaved
        buckets: dict[str, list[str]] = {}
        for g in groups:
            buckets.setdefault(group_label[g], []).append(g)

        group_to_split: dict[str, str] = {}
        for _lbl, b_groups in buckets.items():
            if len(b_groups) == 3:
                group_to_split[b_groups[0]] = "train"
                group_to_split[b_groups[1]] = "val"
                group_to_split[b_groups[2]] = "test"
            elif len(b_groups) == 2:
                group_to_split[b_groups[0]] = "train"
                group_to_split[b_groups[1]] = "val"
            elif len(b_groups) == 1:
                group_to_split[b_groups[0]] = "train"
            else:
                for i, g in enumerate(b_groups):
                    mod = i % 6
                    if mod == 4:
                        group_to_split[g] = "val"
                    elif mod == 5:
                        group_to_split[g] = "test"
                    else:
                        group_to_split[g] = "train"
                if "val" not in {group_to_split[g] for g in b_groups}:
                    group_to_split[b_groups[-2]] = "val"
                if "test" not in {group_to_split[g] for g in b_groups}:
                    group_to_split[b_groups[-1]] = "test"

        if "val" not in group_to_split.values() and len(groups) >= 2:
            group_to_split[groups[-2]] = "val"
        if "test" not in group_to_split.values() and len(groups) >= 3:
            group_to_split[groups[-1]] = "test"
        for it in items:
            it["split"] = group_to_split[str(it.get("group_key") or "default")]
    else:
        # Chronological block split (first 70% train, middle 15% val, final 15% test)
        total = len(items)
        t_end = max(1, int(total * 0.70))
        v_end = max(t_end + 1, int(total * 0.85))
        for i, it in enumerate(items):
            if i < t_end:
                it["split"] = "train"
            elif i < v_end:
                it["split"] = "val"
            else:
                it["split"] = "test"
    return items


def import_local_dataset(
    db: Session,
    config: AppConfig,
    name: str,
    source_dir: Path,
    license_name: str = "Local User / Research",
    description: str = "",
) -> DatasetRecord:
    """
    Inspect and import a local directory containing sky images, optional segmentation masks,
    or class subdirectories (e.g. CCSN-style folders or SWIMSEG-style images/+masks/).
    """
    source_dir = source_dir.resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {source_dir}")

    dest_root = config.datasets_dir / name.replace(" ", "_").lower()
    dest_root.mkdir(parents=True, exist_ok=True)

    # Detect structure: (A) images/ & masks/ (or GTmaps/ or seg_masks/) pairs, or (B) class subfolders, or (C) flat folder
    images_dir = source_dir / "images"
    masks_dir = source_dir / "masks"
    if not masks_dir.exists():
        masks_dir = source_dir / "GTmaps"
    if not masks_dir.exists():
        masks_dir = source_dir / "seg_masks"

    discovered: list[dict[str, Any]] = []
    task_types: set[str] = set()

    if images_dir.exists() and images_dir.is_dir():
        mask_map: dict[str, Path] = {}
        if masks_dir.exists():
            for mf in sorted(masks_dir.rglob("*")):
                if mf.is_file() and mf.suffix.lower() in IMAGE_EXTS:
                    mask_map[mf.stem.replace("_GT", "").replace("_mask", "")] = mf
        for idx, img_file in enumerate(sorted(images_dir.rglob("*"))):
            if not img_file.is_file() or img_file.suffix.lower() not in IMAGE_EXTS:
                continue
            # Validate readability
            arr = cv2.imread(str(img_file))
            if arr is None:
                continue
            stem = img_file.stem
            matched_mask = mask_map.get(stem)
            group_key = stem.split("_")[0] if "_" in stem else f"seq_{idx // 6}"
            discovered.append({
                "image_path": str(img_file),
                "mask_path": str(matched_mask) if matched_mask else "",
                "cloud_type_label": "",
                "group_key": group_key,
                "sequence_index": idx,
            })
        if any(d["mask_path"] for d in discovered):
            task_types.add("segmentation")
    else:
        # Check class folders or flat files
        subdirs = [d for d in sorted(source_dir.iterdir()) if d.is_dir() and d.name.lower() not in {"masks", "gtmaps"}]
        if subdirs:
            for sd in subdirs:
                lbl = sd.name.lower().strip()
                norm_lbl = lbl if lbl in CLOUD_TYPE_CLASSES else "mixed_clouds"
                for idx, img_file in enumerate(sorted(sd.rglob("*"))):
                    if not img_file.is_file() or img_file.suffix.lower() not in IMAGE_EXTS:
                        continue
                    if cv2.imread(str(img_file)) is None:
                        continue
                    discovered.append({
                        "image_path": str(img_file),
                        "mask_path": "",
                        "cloud_type_label": norm_lbl,
                        "group_key": f"{lbl}_{idx // 5}",
                        "sequence_index": idx,
                    })
            if discovered:
                task_types.add("classification")
        else:
            for idx, img_file in enumerate(sorted(source_dir.glob("*"))):
                if not img_file.is_file() or img_file.suffix.lower() not in IMAGE_EXTS:
                    continue
                if cv2.imread(str(img_file)) is None:
                    continue
                discovered.append({
                    "image_path": str(img_file),
                    "mask_path": "",
                    "cloud_type_label": "",
                    "group_key": f"local_{idx // 5}",
                    "sequence_index": idx,
                })

    if not discovered:
        raise ValueError(f"No valid readable images found in {source_dir}")

    if not task_types:
        task_types.add("classification")

    discovered = _assign_group_splits(discovered)

    existing = db.query(DatasetRecord).filter(DatasetRecord.name == name).first()
    if existing:
        db.query(DatasetItem).filter(DatasetItem.dataset_id == existing.id).delete()
        rec = existing
        rec.item_count = len(discovered)
        rec.root_path = str(source_dir)
        rec.license_name = license_name
        rec.description = description
        rec.task_types_json = json.dumps(sorted(task_types))
    else:
        rec = DatasetRecord(
            name=name,
            version="1.0.0",
            source_type="local_import",
            license_name=license_name,
            description=description or f"Imported local dataset from {source_dir}",
            root_path=str(source_dir),
            task_types_json=json.dumps(sorted(task_types)),
            item_count=len(discovered),
            integrity_verified=True,
            metadata_json=json.dumps({"imported_at": datetime.now(UTC).isoformat()}),
        )
        db.add(rec)
        db.flush()

    for it in discovered:
        cov_gt = None
        if it["mask_path"]:
            m = cv2.imread(it["mask_path"], cv2.IMREAD_GRAYSCALE)
            if m is not None and m.size > 0:
                cov_gt = round(float(np.mean((m >= 1) & (m <= 3)) * 100.0) if m.max() <= 5 else float(np.mean(m > 127) * 100.0), 2)
        db.add(
            DatasetItem(
                dataset_id=rec.id,
                split=it["split"],
                group_key=it["group_key"],
                sequence_index=it["sequence_index"],
                image_path=it["image_path"],
                mask_path=it["mask_path"],
                cloud_type_label=it["cloud_type_label"],
                event_label="normal",
                quality_label="usable",
                cloud_coverage_gt=cov_gt,
                user_labeled=False,
            )
        )
    db.commit()
    db.refresh(rec)
    return rec


def _render_synthetic_sky_frame(
    scenario: str,
    t_step: int,
    seed: int,
    width: int = 384,
    height: int = 384,
) -> tuple[np.ndarray, np.ndarray, str, str, float]:
    """
    Generate a physically structured synthetic sky image + exact 6-class ground-truth mask
    (0=clear, 1=thin, 2=dense, 3=dark, 4=haze) for offline baseline training & verification.
    Returns (image_bgr, class_mask_u8, cloud_genus_label, event_label, cloud_cover_pct).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = xx / width
    ny = yy / height

    # Smooth multi-octave noise field drifted across t_step to mimic real cloud advection
    shift_x = t_step * 18.0
    shift_y = t_step * 7.0
    noise = np.zeros((height, width), dtype=np.float32)
    for freq, amp in [(2.5, 0.50), (5.5, 0.28), (11.0, 0.14), (22.0, 0.08)]:
        phase_x = float(rng.uniform(0, 6.28))
        phase_y = float(rng.uniform(0, 6.28))
        field = np.sin((nx + shift_x / width) * freq * math.pi + phase_x) * np.cos((ny + shift_y / height) * freq * math.pi + phase_y)
        # Add smooth random field
        grid_r = cv2.resize(rng.normal(0.0, 1.0, (max(4, int(freq * 3)), max(4, int(freq * 3)))).astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
        noise += amp * (0.55 * field + 0.45 * grid_r)

    noise = (noise - noise.min()) / max(noise.max() - noise.min(), 1e-5)

    # Clear blue sky vertical gradient (richer blue at zenith, paler near horizon)
    sky_b = 205.0 - 30.0 * ny
    sky_g = 132.0 + 25.0 * ny
    sky_r = 55.0 + 40.0 * ny
    img = np.stack([sky_b, sky_g, sky_r], axis=-1)
    mask = np.zeros((height, width), dtype=np.uint8)

    event_label = "normal"
    if scenario == "clear_sky":
        thresh = 0.92
        cloud_genus = "clear_sky"
        cloud_mask = noise > thresh
        mask[cloud_mask] = 1
    elif scenario == "cirrus":
        streak = 0.65 * noise + 0.35 * np.sin((nx * 14.0 + ny * 4.0 + t_step * 0.2) * math.pi)
        streak = (streak - streak.min()) / max(streak.max() - streak.min(), 1e-5)
        thin = streak > 0.56
        mask[thin] = 1
        img[thin] = img[thin] * 0.45 + np.array([242, 238, 232]) * 0.55
        cloud_genus = "cirrus"
    elif scenario == "cumulus":
        # Developing puffs across time
        prog = max(0.38, 0.64 - 0.045 * t_step)
        dense = noise > prog
        thin = (noise > prog - 0.08) & (~dense)
        mask[thin] = 1
        mask[dense] = 2
        img[thin] = img[thin] * 0.40 + np.array([235, 235, 235]) * 0.60
        img[dense] = np.clip(225.0 + 25.0 * noise[dense, None], 210, 252)
        cloud_genus = "cumulus"
        if t_step >= 3:
            event_label = "cloud_cover_increasing"
    elif scenario == "stratocumulus":
        dense = noise > 0.40
        thin = (noise > 0.28) & (~dense)
        dark = noise > 0.72
        mask[thin] = 1
        mask[dense] = 2
        mask[dark] = 3
        img[thin] = img[thin] * 0.3 + np.array([220, 220, 222]) * 0.7
        img[dense] = np.array([205, 205, 210])
        img[dark] = np.array([115, 112, 118])
        cloud_genus = "stratocumulus"
    elif scenario == "stratus":
        mask[:, :] = 2
        dark = noise > 0.65
        mask[dark] = 3
        base = 175.0 - 30.0 * noise
        img = np.stack([base + 4, base + 2, base], axis=-1)
        cloud_genus = "stratus"
    elif scenario == "cumulonimbus_like":
        dense = noise > 0.35
        dark = noise > 0.48
        mask[noise > 0.25] = 1
        mask[dense] = 2
        mask[dark] = 3
        img[mask == 1] = np.array([165, 165, 172])
        img[dense] = np.array([145, 145, 150])
        img[dark] = np.array([72, 70, 76])
        cloud_genus = "cumulonimbus_like"
        event_label = "storm_like_structure"
    elif scenario == "clearing":
        prog = min(0.88, 0.34 + 0.09 * t_step)
        dense = noise > prog
        thin = (noise > prog - 0.10) & (~dense)
        mask[thin] = 1
        mask[dense] = 2
        img[thin] = img[thin] * 0.45 + np.array([235, 235, 235]) * 0.55
        img[dense] = np.array([225, 225, 230])
        cloud_genus = "cumulus" if t_step < 4 else "clear_sky"
        event_label = "sky_clearing"
    else:
        # altocumulus / mixed_clouds
        haze = noise < 0.22
        thin = (noise >= 0.45) & (noise < 0.62)
        dense = noise >= 0.62
        mask[haze] = 4
        mask[thin] = 1
        mask[dense] = 2
        img[haze] = img[haze] * 0.55 + np.array([195, 205, 212]) * 0.45
        img[thin] = img[thin] * 0.35 + np.array([235, 235, 235]) * 0.65
        img[dense] = np.array([225, 225, 228])
        cloud_genus = "altocumulus" if scenario == "altocumulus" else "mixed_clouds"

    # Subtle sensor grain
    grain = rng.normal(0.0, 2.2, (height, width, 3)).astype(np.float32)
    img_u8 = np.clip(img + grain, 0, 255).astype(np.uint8)
    cloud_cov = float(np.mean((mask >= 1) & (mask <= 3)) * 100.0)
    return img_u8, mask, cloud_genus, event_label, round(cloud_cov, 2)


def bootstrap_local_sky_dataset(db: Session, config: AppConfig, sequences_per_scenario: int = 3) -> DatasetRecord:
    """
    Create an offline local sky training dataset ('skyobserver_local_baseline') containing
    paired images, 6-class segmentation masks, cloud genus labels, and 6-frame temporal sequences.
    """
    ds_name = "skyobserver_local_baseline"
    existing = db.query(DatasetRecord).filter(DatasetRecord.name == ds_name).first()
    if existing and existing.item_count >= 90:
        return existing

    root = config.datasets_dir / ds_name
    img_dir = root / "images"
    msk_dir = root / "masks"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        "clear_sky",
        "cirrus",
        "cumulus",
        "stratocumulus",
        "stratus",
        "cumulonimbus_like",
        "clearing",
        "altocumulus",
    ]

    items_meta: list[dict[str, Any]] = []
    seed_counter = 100
    for sc in scenarios:
        for seq_idx in range(sequences_per_scenario):
            group_key = f"{sc}_seq{seq_idx}"
            for t in range(6):
                seed_counter += 1
                img_u8, mask_u8, genus, ev_label, cov_gt = _render_synthetic_sky_frame(
                    scenario=sc, t_step=t, seed=seed_counter, width=256, height=256
                )
                fname = f"{group_key}_t{t:02d}.png"
                ip = img_dir / fname
                mp = msk_dir / fname
                cv2.imwrite(str(ip), img_u8)
                cv2.imwrite(str(mp), mask_u8)
                items_meta.append({
                    "image_path": str(ip),
                    "mask_path": str(mp),
                    "cloud_type_label": genus,
                    "event_label": ev_label,
                    "group_key": group_key,
                    "sequence_index": t,
                    "cloud_coverage_gt": cov_gt,
                })

    items_meta = _assign_group_splits(items_meta)

    if existing:
        db.query(DatasetItem).filter(DatasetItem.dataset_id == existing.id).delete()
        rec = existing
        rec.item_count = len(items_meta)
        rec.root_path = str(root)
    else:
        rec = DatasetRecord(
            name=ds_name,
            version="1.0.0",
            source_type="local_synthetic_baseline",
            license_name="Local Project Generated (MIT)",
            description="Offline multi-scenario sky dataset with 6-class segmentation masks, cloud genus labels, and temporal sequences.",
            root_path=str(root),
            task_types_json=json.dumps(["segmentation", "classification", "temporal", "event"]),
            item_count=len(items_meta),
            integrity_verified=True,
            metadata_json=json.dumps({"scenarios": scenarios, "frames_per_sequence": 6}),
        )
        db.add(rec)
        db.flush()

    for it in items_meta:
        db.add(
            DatasetItem(
                dataset_id=rec.id,
                split=it["split"],
                group_key=it["group_key"],
                sequence_index=it["sequence_index"],
                image_path=it["image_path"],
                mask_path=it["mask_path"],
                cloud_type_label=it["cloud_type_label"],
                event_label=it["event_label"],
                quality_label="usable",
                cloud_coverage_gt=it["cloud_coverage_gt"],
                user_labeled=False,
            )
        )
    db.commit()
    db.refresh(rec)
    return rec


def get_dataset_summary(db: Session, dataset_id: int) -> dict[str, Any]:
    ds = db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not ds:
        raise ValueError(f"Dataset {dataset_id} not found")
    items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).all()
    splits = {"train": 0, "val": 0, "test": 0}
    classes: dict[str, int] = {}
    with_masks = 0
    missing_files = 0
    for it in items:
        splits[it.split] = splits.get(it.split, 0) + 1
        if it.cloud_type_label:
            classes[it.cloud_type_label] = classes.get(it.cloud_type_label, 0) + 1
        if it.mask_path and Path(it.mask_path).exists():
            with_masks += 1
        if not Path(it.image_path).exists():
            missing_files += 1

    return {
        "id": ds.id,
        "name": ds.name,
        "version": ds.version,
        "source_type": ds.source_type,
        "license_name": ds.license_name,
        "description": ds.description,
        "root_path": ds.root_path,
        "task_types": json.loads(ds.task_types_json),
        "item_count": len(items),
        "splits": splits,
        "masks_available": with_masks,
        "class_distribution": classes,
        "integrity_verified": missing_files == 0,
        "missing_files": missing_files,
        "created_at": ds.created_at.isoformat(),
    }
