"""7.5 Mask-Aware Cloud Motion Tracking module (Farneback, Lucas-Kanade, Phase Correlation)."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from app.vision.geometry import CameraGeometry, project_pixels_to_cloud_plane


def _deg_to_cardinal(deg: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int(((deg + 22.5) % 360.0) // 45.0)
    return dirs[idx]


def _mad_inliers(vx: np.ndarray, vy: np.ndarray, z_thresh: float = 3.0) -> np.ndarray:
    """Return boolean mask of flow vectors within z_thresh MAD of the 2D median."""
    if len(vx) < 8:
        return np.ones(len(vx), dtype=bool)
    mx, my = float(np.median(vx)), float(np.median(vy))
    dist = np.hypot(vx - mx, vy - my)
    mad = float(np.median(dist))
    scale = max(mad * 1.4826, 0.35)
    return dist <= (z_thresh * scale)


def _detect_secondary_layer(
    fx: np.ndarray,
    fy: np.ndarray,
    primary_dx: float,
    primary_dy: float,
    consistency: float,
) -> dict[str, Any]:
    """Detect a distinct secondary cloud deck velocity when flow is multi-modal."""
    if len(fx) < 36 or consistency >= 0.78:
        return {"available": False}
    mags = np.hypot(fx, fy)
    active = mags > 0.3
    if int(active.sum()) < 24:
        return {"available": False}
    afx, afy = fx[active], fy[active]
    pmag = max(math.hypot(primary_dx, primary_dy), 1e-4)
    #Vectors differing strongly in direction (>40 deg) or speed (>1.7x) from primary
    cos_p = (afx * primary_dx + afy * primary_dy) / (np.hypot(afx, afy) * pmag)
    speed_ratio = np.hypot(afx, afy) / pmag
    outliers = (cos_p < 0.766) | (speed_ratio > 1.75) | (speed_ratio < 0.45)
    share = float(np.mean(outliers))
    if int(np.count_nonzero(outliers)) < 12 or share < 0.15 or share > 0.65:
        return {"available": False}
    sdx = float(np.median(afx[outliers]))
    sdy = float(np.median(afy[outliers]))
    smag = float(math.hypot(sdx, sdy))
    if smag < 0.3:
        return {"available": False}
    sdir = float((math.degrees(math.atan2(sdx, -sdy)) + 360.0) % 360.0)
    return {
        "available": True,
        "dx_px_s": round(sdx, 3),
        "dy_px_s": round(sdy, 3),
        "magnitude_px_s": round(smag, 3),
        "direction_deg": round(sdir, 1),
        "direction_cardinal": _deg_to_cardinal(sdir),
        "share_pct": round(share * 100.0, 1),
    }


def estimate_cloud_motion(
    prev_bgr: np.ndarray | None,
    curr_bgr: np.ndarray,
    valid_mask: np.ndarray,
    cloud_binary_mask: np.ndarray,
    dt_seconds: float,
    fov_horizontal_deg: float | None = 72.0,
    assumed_cloud_altitude_m: float | None = None,
    grid_rows: int = 3,
    grid_cols: int = 3,
    fov_vertical_deg: float | None = None,
    orientation_elevation_deg: float = 90.0,
    orientation_azimuth_deg: float = 0.0,
    orientation_roll_deg: float = 0.0,
    tilt_correction_enabled: bool = False,
) -> tuple[dict[str, Any], np.ndarray]:
    """
    Estimate cloud movement vectors using Farneback optical flow, Lucas-Kanade sparse flow,
    and phase correlation strictly inside valid sky/cloud pixels.

    Returns (motion_metrics_dict, motion_overlay_bgr).
    """
    orig_h, orig_w = curr_bgr.shape[:2]
    overlay = curr_bgr.copy()
    valid_bool = valid_mask > 0
    overlay[~valid_bool] = (overlay[~valid_bool] * 0.35).astype(np.uint8)

    if prev_bgr is None or prev_bgr.shape[:2] != curr_bgr.shape[:2] or dt_seconds <= 0.05:
        empty = {
            "available": False,
            "reason": "Need at least two sequential frames with positive time delta",
            "dx_px_s": 0.0,
            "dy_px_s": 0.0,
            "mean_vector": [0.0, 0.0],
            "median_vector": [0.0, 0.0],
            "magnitude_px_s": 0.0,
            "angular_speed_deg_s": None,
            "derived_ground_speed_kmh": None,
            "derived_speed_note": "Requires two consecutive frames",
            "direction_deg": 0.0,
            "direction_cardinal": "N",
            "consistency": 0.0,
            "confidence": 0.0,
            "phase_correlation_vector": [0.0, 0.0],
            "lucas_kanade_vector": [0.0, 0.0],
            "regional_vectors": [],
        }
        return empty, overlay

    work_w, work_h = 480, 360
    scale_x = orig_w / float(work_w)
    scale_y = orig_h / float(work_h)

    prev_g = cv2.cvtColor(cv2.resize(prev_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(cv2.resize(curr_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    m_work = cv2.resize(valid_mask, (work_w, work_h), interpolation=cv2.INTER_NEAREST) > 0
    if not np.all(m_work) and int(m_work.sum()) > 200:
        eroded = cv2.erode(m_work.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
        if int(eroded.sum()) > 120:
            m_work = eroded
    c_work = (cv2.resize(cloud_binary_mask, (work_w, work_h), interpolation=cv2.INTER_NEAREST) > 0) & m_work

    # Fill excluded pixels (logos, roads, trees, waves) with valid-sky mean gray so edges do not bleed
    if not np.all(m_work) and np.any(m_work):
        prev_g = prev_g.copy()
        curr_g = curr_g.copy()
        prev_g[~m_work] = int(np.mean(prev_g[m_work]))
        curr_g[~m_work] = int(np.mean(curr_g[m_work]))

    # Mild CLAHE stabilizes flow against auto-exposure flicker between sequential frames
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    prev_g = clahe.apply(prev_g)
    curr_g = clahe.apply(curr_g)

    # Prefer cloud pixels when enough cloud area exists; fallback to valid sky pixels
    roi_mask = c_work if int(c_work.sum()) > 180 else m_work
    if int(roi_mask.sum()) < 60:
        roi_mask = np.ones((work_h, work_w), dtype=bool)

    # 1. Farneback dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_g,
        curr_g,
        None,
        pyr_scale=0.5,
        levels=4,
        winsize=17,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    fx_raw = flow[:, :, 0][roi_mask] * scale_x / dt_seconds
    fy_raw = flow[:, :, 1][roi_mask] * scale_y / dt_seconds
    inliers = _mad_inliers(fx_raw, fy_raw, z_thresh=3.0)
    fx = fx_raw[inliers] if int(np.count_nonzero(inliers)) >= 10 else fx_raw
    fy = fy_raw[inliers] if int(np.count_nonzero(inliers)) >= 10 else fy_raw

    mean_dx = float(np.mean(fx))
    mean_dy = float(np.mean(fy))
    med_dx = float(np.median(fx))
    med_dy = float(np.median(fy))

    # 2. Lucas-Kanade sparse flow on cloud texture features with MAD outlier rejection
    lk_dx, lk_dy = 0.0, 0.0
    lk_valid_count = 0
    p0 = cv2.goodFeaturesToTrack(
        prev_g,
        maxCorners=100,
        qualityLevel=0.02,
        minDistance=10,
        mask=roi_mask.astype(np.uint8) * 255,
    )
    if p0 is not None and len(p0) > 0:
        p1, st, _err = cv2.calcOpticalFlowPyrLK(prev_g, curr_g, p0, None, winSize=(19, 19), maxLevel=3)
        if p1 is not None and st is not None:
            good_new = p1[st.flatten() == 1]
            good_old = p0[st.flatten() == 1]
            if len(good_new) > 0:
                diffs = (good_new - good_old).reshape(-1, 2)
                lk_in = _mad_inliers(diffs[:, 0], diffs[:, 1], z_thresh=2.8)
                if int(np.count_nonzero(lk_in)) >= 4:
                    diffs = diffs[lk_in]
                    good_new = good_new[lk_in]
                    good_old = good_old[lk_in]
                lk_dx = float(np.median(diffs[:, 0]) * scale_x / dt_seconds)
                lk_dy = float(np.median(diffs[:, 1]) * scale_y / dt_seconds)
                lk_valid_count = len(good_new)

    # 3. Masked phase correlation
    prev_masked = (prev_g.astype(np.float32) - np.mean(prev_g[roi_mask])) * roi_mask.astype(np.float32)
    curr_masked = (curr_g.astype(np.float32) - np.mean(curr_g[roi_mask])) * roi_mask.astype(np.float32)
    (pc_x, pc_y), pc_resp = cv2.phaseCorrelate(prev_masked, curr_masked)
    pc_dx = float(pc_x * scale_x / dt_seconds)
    pc_dy = float(pc_y * scale_y / dt_seconds)

    # Combine median Farneback with LK when available
    lk_pts_old = None
    lk_pts_new = None
    if p0 is not None and len(p0) > 0 and lk_valid_count > 0:
        lk_pts_old = good_old.reshape(-1, 2)
        lk_pts_new = good_new.reshape(-1, 2)

    if lk_valid_count >= 5:
        best_dx = 0.7 * med_dx + 0.3 * lk_dx
        best_dy = 0.7 * med_dy + 0.3 * lk_dy
    else:
        best_dx = med_dx
        best_dy = med_dy

    magnitude_px_s = float(np.hypot(best_dx, best_dy))

    # Direction in image-compass convention (0 deg = Up/North, 90 = Right/East, 180 = Down/South, 270 = Left/West)
    raw_dir_deg = float((np.degrees(np.arctan2(best_dx, -best_dy)) + 360.0) % 360.0) if magnitude_px_s > 0.05 else 0.0
    direction_deg = raw_dir_deg
    cardinal = _deg_to_cardinal(direction_deg)

    # Circular directional consistency of flow vectors
    mags = np.hypot(fx, fy)
    active = mags > 0.25
    if int(active.sum()) >= 10:
        ux = fx[active] / mags[active]
        uy = fy[active] / mags[active]
        consistency = float(np.hypot(np.mean(ux), np.mean(uy)))
    else:
        consistency = 0.0

    # Confidence based on cloud availability, directional consistency, and Farneback-LK/PC agreement
    cloud_factor = min(1.0, float(c_work.sum()) / 1500.0)
    agreement = 1.0
    if lk_valid_count >= 5 and magnitude_px_s > 0.2:
        cos_sim = (med_dx * lk_dx + med_dy * lk_dy) / (max(np.hypot(med_dx, med_dy), 1e-4) * max(np.hypot(lk_dx, lk_dy), 1e-4))
        agreement = float(max(0.0, (cos_sim + 1.0) * 0.5))
    confidence = float(np.clip(0.55 * consistency + 0.25 * cloud_factor + 0.20 * agreement, 0.0, 1.0))

    # Angular speed if horizontal FOV is provided
    angular_speed_deg_s: float | None = None
    derived_kmh: float | None = None
    speed_note = "Pixel motion only; configure camera FOV for angular speed."
    if fov_horizontal_deg is not None and fov_horizontal_deg > 0:
        deg_per_px = fov_horizontal_deg / float(orig_w)
        angular_speed_deg_s = round(magnitude_px_s * deg_per_px, 4)
        speed_note = "Angular velocity computed from configured camera FOV; altitude unknown."
        if assumed_cloud_altitude_m is not None and assumed_cloud_altitude_m > 0:
            rad_s = np.radians(angular_speed_deg_s)
            derived_kmh = round(float(rad_s * assumed_cloud_altitude_m * 3.6), 2)
            speed_note = f"Derived estimate assuming cloud base altitude {assumed_cloud_altitude_m:.0f} m (unverified)."

    # Tilt-corrected planar projection onto horizontal cloud deck
    tilt_applied = False
    world_ve_norm_s: float | None = None
    world_vn_norm_s: float | None = None
    geom: CameraGeometry | None = None
    if tilt_correction_enabled:
        fov_h = float(fov_horizontal_deg) if (fov_horizontal_deg and fov_horizontal_deg > 0) else 72.0
        if fov_vertical_deg is not None and fov_vertical_deg > 0:
            fov_v = float(fov_vertical_deg)
        else:
            fov_v = 2.0 * math.degrees(math.atan(math.tan(math.radians(fov_h) / 2.0) * orig_h / float(max(orig_w, 1))))
        geom = CameraGeometry(
            elevation_deg=max(-20.0, min(90.0, float(orientation_elevation_deg))),
            azimuth_deg=float(orientation_azimuth_deg) % 360.0,
            roll_deg=max(-45.0, min(45.0, float(orientation_roll_deg))),
            fov_horizontal_deg=max(10.0, min(170.0, fov_h)),
            fov_vertical_deg=max(10.0, min(170.0, fov_v)),
            cloud_altitude_m=float(assumed_cloud_altitude_m) if (assumed_cloud_altitude_m and assumed_cloud_altitude_m > 0) else 2000.0,
            tilt_correction_enabled=True,
        )
        vy_idx, vx_idx = np.where(roi_mask)
        u0 = vx_idx.astype(np.float64) * scale_x
        v0 = vy_idx.astype(np.float64) * scale_y
        u1 = u0 + flow[roi_mask, 0].astype(np.float64) * scale_x
        v1 = v0 + flow[roi_mask, 1].astype(np.float64) * scale_y
        e1, n1, ok1 = project_pixels_to_cloud_plane(u0, v0, orig_w, orig_h, geom, altitude_m=1.0)
        e2, n2, ok2 = project_pixels_to_cloud_plane(u1, v1, orig_w, orig_h, geom, altitude_m=1.0)
        both_ok = ok1 & ok2
        if int(np.count_nonzero(both_ok)) >= 10:
            ve_arr = (e2[both_ok] - e1[both_ok]) / dt_seconds
            vn_arr = (n2[both_ok] - n1[both_ok]) / dt_seconds
            med_ve = float(np.median(ve_arr))
            med_vn = float(np.median(vn_arr))
            best_ve, best_vn = med_ve, med_vn

            lk_proj_ok = False
            lk_ve, lk_vn = 0.0, 0.0
            if lk_pts_old is not None and lk_pts_new is not None and lk_valid_count >= 5:
                le1, ln1, lok1 = project_pixels_to_cloud_plane(
                    lk_pts_old[:, 0] * scale_x, lk_pts_old[:, 1] * scale_y, orig_w, orig_h, geom, altitude_m=1.0
                )
                le2, ln2, lok2 = project_pixels_to_cloud_plane(
                    lk_pts_new[:, 0] * scale_x, lk_pts_new[:, 1] * scale_y, orig_w, orig_h, geom, altitude_m=1.0
                )
                lok = lok1 & lok2
                if int(np.count_nonzero(lok)) >= 5:
                    lk_ve = float(np.median((le2[lok] - le1[lok]) / dt_seconds))
                    lk_vn = float(np.median((ln2[lok] - ln1[lok]) / dt_seconds))
                    best_ve = 0.7 * med_ve + 0.3 * lk_ve
                    best_vn = 0.7 * med_vn + 0.3 * lk_vn
                    lk_proj_ok = True

            world_mag_norm = float(math.hypot(best_ve, best_vn))
            world_ve_norm_s = round(best_ve, 6)
            world_vn_norm_s = round(best_vn, 6)
            tilt_applied = True
            direction_deg = (
                float((math.degrees(math.atan2(best_ve, best_vn)) + 360.0) % 360.0)
                if (magnitude_px_s > 0.05 and world_mag_norm > 1e-5)
                else 0.0
            )
            cardinal = _deg_to_cardinal(direction_deg)
            angular_speed_deg_s = round(float(math.degrees(world_mag_norm)), 4)

            wmags = np.hypot(ve_arr, vn_arr)
            wact = wmags > 1e-4
            if int(np.count_nonzero(wact)) >= 10:
                consistency = float(np.hypot(np.mean(ve_arr[wact] / wmags[wact]), np.mean(vn_arr[wact] / wmags[wact])))
            else:
                consistency = 0.0
            w_agree = 1.0
            if lk_proj_ok and world_mag_norm > 1e-4:
                cos_w = (med_ve * lk_ve + med_vn * lk_vn) / (
                    max(math.hypot(med_ve, med_vn), 1e-6) * max(math.hypot(lk_ve, lk_vn), 1e-6)
                )
                w_agree = float(max(0.0, (cos_w + 1.0) * 0.5))
            confidence = float(np.clip(0.55 * consistency + 0.25 * cloud_factor + 0.20 * w_agree, 0.0, 1.0))

            if assumed_cloud_altitude_m is not None and assumed_cloud_altitude_m > 0:
                derived_kmh = round(float(world_mag_norm * assumed_cloud_altitude_m * 3.6), 2)
                speed_note = (
                    f"Tilt-corrected ({geom.elevation_deg:.0f}° el, {geom.azimuth_deg:.0f}° az) at {assumed_cloud_altitude_m:.0f} m."
                )
            else:
                derived_kmh = None
                speed_note = f"Tilt-corrected ({geom.elevation_deg:.0f}° el, {geom.azimuth_deg:.0f}° az); cloud altitude unset."

    # Regional motion vectors across grid
    regional_vectors: list[dict[str, Any]] = []
    rh = work_h // grid_rows
    cw = work_w // grid_cols
    for r in range(grid_rows):
        for c in range(grid_cols):
            y0, y1 = r * rh, (r + 1) * rh if r < grid_rows - 1 else work_h
            x0, x1 = c * cw, (c + 1) * cw if c < grid_cols - 1 else work_w
            cm = roi_mask[y0:y1, x0:x1]
            if int(cm.sum()) > 10:
                cdx = float(np.median(flow[y0:y1, x0:x1, 0][cm]) * scale_x / dt_seconds)
                cdy = float(np.median(flow[y0:y1, x0:x1, 1][cm]) * scale_y / dt_seconds)
            else:
                cdx, cdy = 0.0, 0.0
            cmag = float(np.hypot(cdx, cdy))
            cdir = float((np.degrees(np.arctan2(cdx, -cdy)) + 360.0) % 360.0) if cmag > 0.05 else 0.0
            if tilt_applied and geom is not None and cmag > 0.05:
                cy_mid = (y0 + y1) * 0.5 * scale_y
                cx_mid = (x0 + x1) * 0.5 * scale_x
                ce1, cn1, cok1 = project_pixels_to_cloud_plane(
                    np.array([cx_mid]), np.array([cy_mid]), orig_w, orig_h, geom, altitude_m=1.0
                )
                ce2, cn2, cok2 = project_pixels_to_cloud_plane(
                    np.array([cx_mid + cdx * dt_seconds]),
                    np.array([cy_mid + cdy * dt_seconds]),
                    orig_w,
                    orig_h,
                    geom,
                    altitude_m=1.0,
                )
                if bool(cok1[0] and cok2[0]):
                    cve = float((ce2[0] - ce1[0]) / dt_seconds)
                    cvn = float((cn2[0] - cn1[0]) / dt_seconds)
                    if math.hypot(cve, cvn) > 1e-5:
                        cdir = float((math.degrees(math.atan2(cve, cvn)) + 360.0) % 360.0)
            regional_vectors.append({
                "row": r,
                "col": c,
                "dx_px_s": round(cdx, 2),
                "dy_px_s": round(cdy, 2),
                "magnitude_px_s": round(cmag, 2),
                "direction_deg": round(cdir, 1),
            })

    # Draw movement vector arrows on overlay
    step_y = max(16, orig_h // 12)
    step_x = max(16, orig_w // 16)
    for oy in range(step_y // 2, orig_h, step_y):
        for ox in range(step_x // 2, orig_w, step_x):
            wy = min(work_h - 1, int(oy / scale_y))
            wx = min(work_w - 1, int(ox / scale_x))
            if not roi_mask[wy, wx]:
                continue
            vx = float(flow[wy, wx, 0] * scale_x)
            vy = float(flow[wy, wx, 1] * scale_y)
            mag = float(np.hypot(vx, vy))
            if mag < 0.4:
                cv2.circle(overlay, (ox, oy), 2, (120, 220, 255), -1)
            else:
                draw_scale = min(45.0, max(10.0, mag * 3.5)) / mag
                end_pt = (int(ox + vx * draw_scale), int(oy + vy * draw_scale))
                cv2.arrowedLine(overlay, (ox, oy), end_pt, (0, 235, 255), 2, tipLength=0.35)

    metrics = {
        "available": True,
        "dx_px_s": round(best_dx, 3),
        "dy_px_s": round(best_dy, 3),
        "mean_vector": [round(mean_dx, 3), round(mean_dy, 3)],
        "median_vector": [round(med_dx, 3), round(med_dy, 3)],
        "magnitude_px_s": round(magnitude_px_s, 3),
        "angular_speed_deg_s": angular_speed_deg_s,
        "derived_ground_speed_kmh": derived_kmh,
        "derived_speed_note": speed_note,
        "direction_deg": round(direction_deg, 1),
        "direction_cardinal": cardinal,
        "raw_image_direction_deg": round(raw_dir_deg, 1),
        "tilt_corrected": tilt_applied,
        "camera_elevation_deg": round(float(orientation_elevation_deg), 2) if tilt_applied else None,
        "camera_azimuth_deg": round(float(orientation_azimuth_deg) % 360.0, 2) if tilt_applied else None,
        "world_ve_norm_s": world_ve_norm_s,
        "world_vn_norm_s": world_vn_norm_s,
        "consistency": round(consistency, 3),
        "confidence": round(confidence, 3),
        "phase_correlation_vector": [round(pc_dx, 3), round(pc_dy, 3)],
        "phase_correlation_response": round(float(pc_resp), 3),
        "lucas_kanade_vector": [round(lk_dx, 3), round(lk_dy, 3)],
        "lucas_kanade_tracks": lk_valid_count,
        "secondary_layer": _detect_secondary_layer(fx_raw, fy_raw, best_dx, best_dy, consistency),
        "regional_vectors": regional_vectors,
    }
    return metrics, overlay
