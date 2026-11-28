"""Local Weather Nowcasting Engine combining transparent rule-based signals and TemporalSkyConvGRU."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from sqlalchemy.orm import Session

from app.ai.registry import get_model_cache
from app.config import AppConfig
from app.forecast.advection import advect_cloud_mask_optical_flow, persistence_baseline_cover
from app.forecast.verification import compute_verification_summary
from app.models_db import Forecast, Observation, safe_json_loads
from app.services.weather_online import get_online_weather_service
from app.vision.masking import build_valid_sky_mask_from_config


def _classify_cover_condition(cover_pct: float, dark_pct: float, storm_prob: float) -> str:
    if storm_prob >= 0.60 or dark_pct >= 40.0:
        return "storm_like_visual_development"
    if cover_pct < 15.0:
        return "clear"
    if cover_pct < 35.0:
        return "mostly_clear"
    if cover_pct < 65.0:
        return "partly_cloudy"
    if cover_pct < 82.0:
        return "mostly_cloudy"
    if dark_pct >= 25.0:
        return "dark_overcast"
    return "overcast"


def _obs_to_temporal_vector(obs: Observation) -> np.ndarray:
    col = safe_json_loads(obs.color_metrics_json, {})
    tex = safe_json_loads(obs.texture_metrics_json, {})
    pat = col.get("pattern_ratios", {})
    return np.array(
        [
            obs.cloud_coverage_pct / 100.0,
            obs.clear_sky_pct / 100.0,
            obs.thin_cloud_pct / 100.0,
            obs.dense_cloud_pct / 100.0,
            obs.dark_cloud_pct / 100.0,
            obs.haze_fog_pct / 100.0,
            obs.avg_luminance / 100.0,
            obs.blue_sky_index,
            float(col.get("saturation", 0.2)),
            float(col.get("grayness", 0.4)),
            float(pat.get("blue_sky", 0.3)),
            float(pat.get("white_cloud", 0.2)),
            float(pat.get("dark_cloud", 0.1)),
            float(tex.get("edge_density", 0.03)) * 5.0,
            float(tex.get("local_contrast", 10.0)) / 40.0,
            float(tex.get("fragmentation_index", 0.2)),
            float(tex.get("glcm_contrast", 2.0)) / 20.0,
            float(tex.get("glcm_homogeneity", 0.6)),
            float(tex.get("glcm_entropy", 3.0)) / 8.0,
            obs.cloud_coverage_pct / 100.0,
        ],
        dtype=np.float32,
    )


def generate_forecasts_for_observation(
    db: Session,
    latest_obs: Observation,
    config: AppConfig,
) -> list[Forecast]:
    """
    Generate image-only short-term forecasts (5, 15, 30, 60 min) using strictly local observations.
    Refuses to fabricate forecasts if frame quality is unusable.
    """
    obs_time = latest_obs.captured_at or datetime.now(UTC)
    if obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=UTC)
    gen_time = datetime.now(UTC)

    # Replace any prior unverified forecasts generated for this exact observation (e.g. mode/location toggle)
    db.query(Forecast).filter(
        Forecast.observation_id == latest_obs.id,
        Forecast.verified.is_(False),
    ).delete(synchronize_session=False)

    limitations = [
        "image_only_prediction",
        "no_barometric_or_humidity_sensor",
        "no_rain_gauge",
        "no_external_weather_data",
    ]

    if not latest_obs.is_usable:
        fc = Forecast(
            generated_at=gen_time,
            observation_id=latest_obs.id,
            horizon_minutes=15,
            target_time=obs_time + timedelta(minutes=15),
            status="unavailable",
            forecast_condition="unavailable_low_quality_frame",
            cloud_cover_prediction=0.0,
            confidence=0.0,
            signals_json=json.dumps(["frame_flagged_unusable"]),
            reasons_json=json.dumps(["Latest camera frame failed image quality verification."]),
            limitations_json=json.dumps(limitations),
            model_version="unavailable",
            verified=True,
        )
        db.add(fc)
        db.commit()
        return [fc]

    recent_obs = (
        db.query(Observation)
        .filter(Observation.is_usable.is_(True), Observation.id <= latest_obs.id)
        .order_by(Observation.captured_at.desc(), Observation.id.desc())
        .limit(8)
        .all()
    )
    recent_obs.reverse()  # oldest -> newest

    cov_curr = float(latest_obs.cloud_coverage_pct)
    dark_curr = float(latest_obs.dark_cloud_pct)
    dense_curr = float(latest_obs.dense_cloud_pct)
    lum_curr = float(latest_obs.avg_luminance)
    cov_meta = safe_json_loads(latest_obs.coverage_metrics_json, {})
    tex_curr = safe_json_loads(latest_obs.texture_metrics_json, {})
    cls_probs = safe_json_loads(latest_obs.classification_probs_json, {})

    rate_per_min = float(cov_meta.get("rate_of_change_pct_per_min", 0.0))
    lum_delta = 0.0
    dense_delta = 0.0
    dark_delta = 0.0
    if len(recent_obs) >= 2:
        prev = recent_obs[0]
        lum_delta = lum_curr - float(prev.avg_luminance)
        dense_delta = dense_curr - float(prev.dense_cloud_pct)
        dark_delta = dark_curr - float(prev.dark_cloud_pct)

    rules = config.forecast.rules
    signals: list[str] = []
    reasons: list[str] = []

    if rate_per_min >= rules.rapid_increase_rate_pct_per_min:
        signals.append("cloud_cover_rapidly_increasing")
        reasons.append(f"Cloud coverage rising at {rate_per_min:+.2f}%/min over recent frames.")
    elif rate_per_min > 0.3:
        signals.append("cloud_cover_increasing")
        reasons.append(f"Gradual cloud expansion detected ({rate_per_min:+.2f}%/min).")
    elif rate_per_min <= rules.rapid_decrease_rate_pct_per_min:
        signals.append("cloud_cover_rapidly_decreasing")
        reasons.append(f"Sky clearing rapidly at {rate_per_min:+.2f}%/min.")
    elif rate_per_min < -0.3:
        signals.append("cloud_cover_decreasing")
        reasons.append(f"Cloud fraction receding ({rate_per_min:+.2f}%/min).")
    else:
        signals.append("cloud_cover_relatively_steady")
        reasons.append("Cloud coverage rate of change is currently steady.")

    if lum_delta <= rules.darkening_luminance_delta:
        signals.append("brightness_decreasing")
        reasons.append(f"Sky luminance dropped by {abs(lum_delta):.1f} L* units across recent captures.")
    if dense_delta > 6.0:
        signals.append("dense_cloud_area_expanding")
        reasons.append(f"Dense cloud proportion grew by +{dense_delta:.1f}%.")
    if dark_delta > 5.0 or dark_curr >= rules.storm_dark_cloud_pct:
        signals.append("dark_cloud_area_increasing")
        reasons.append(f"Dark cloud base fraction at {dark_curr:.1f}%.")
    if latest_obs.motion_consistency >= rules.high_motion_consistency and latest_obs.motion_speed_px_s > 0.5:
        signals.append("cloud_movement_consistent")
        reasons.append(
            f"Consistent cloud advection toward {latest_obs.motion_direction_cardinal} "
            f"({latest_obs.motion_speed_px_s:.1f} px/s)."
        )
    if float(tex_curr.get("fragmentation_index", 0.0)) > 0.40:
        signals.append("sky_fragmented_cellular")
    elif cov_curr > 75.0 and float(tex_curr.get("largest_fragment_ratio", 0.0)) > 0.75:
        signals.append("uniform_cloud_layer_developing")

    # Optional TemporalSkyConvGRU model inference
    temp_model, temp_meta, temp_dev = get_model_cache().get("temporal")
    nn_horizon_preds: np.ndarray | None = None
    model_version = "rules-v1.0.0"
    if temp_model is not None and len(recent_obs) >= 2:
        vecs = [_obs_to_temporal_vector(o) for o in recent_obs[-4:]]
        while len(vecs) < 4:
            vecs.insert(0, vecs[0])
        seq_t = torch.tensor(np.stack(vecs, axis=0), dtype=torch.float32).unsqueeze(0).to(temp_dev)
        temp_model.eval()
        with torch.no_grad():
            raw_out = temp_model(seq_t).squeeze(0).cpu().numpy()  # (4, 6)
        nn_horizon_preds = raw_out
        model_version = f"hybrid-{temp_meta.get('version', 'v1')}-rules"

    # Optional Online Weather API Fusion (only when offline_mode is explicitly False)
    online_data: dict[str, Any] | None = None
    if not config.server.offline_mode:
        wx = get_online_weather_service().get_weather(config.camera.latitude, config.camera.longitude, config)
        if wx.get("available"):
            online_data = wx
            cur_wx = wx.get("current", {})
            signals.append("open_meteo_nwp_fusion")
            reasons.append(
                f"Online Open-Meteo sync ({config.camera.latitude:.2f}, {config.camera.longitude:.2f}): "
                f"{cur_wx.get('temperature_c', '?')}°C, {cur_wx.get('relative_humidity_pct', '?')}% RH, "
                f"{cur_wx.get('pressure_msl_hpa', '?')} hPa, wind {cur_wx.get('wind_speed_kmh', '?')} km/h."
            )
            if (cur_wx.get("pressure_msl_hpa") or 1015) < 1008:
                signals.append("open_meteo_low_pressure")
            if (cur_wx.get("relative_humidity_pct") or 0) >= 82:
                signals.append("open_meteo_high_humidity")
            model_version = f"{model_version}+open-meteo"
            limitations = [
                "hybrid_local_camera_vision_and_open_meteo_nwp",
                "regional_nwp_resolution_approx_2_to_11km",
            ]

    # Load latest segmentation mask & optical-flow vector for pixel-level advection baseline + ensemble
    seg_mask: np.ndarray | None = None
    valid_mask_adv: np.ndarray | None = None
    motion_meta = safe_json_loads(latest_obs.motion_metrics_json, {})
    dx_px_s = float(motion_meta.get("dx_px_s", 0.0) or 0.0)
    dy_px_s = float(motion_meta.get("dy_px_s", 0.0) or 0.0)
    if latest_obs.seg_mask_path and Path(latest_obs.seg_mask_path).exists():
        loaded_mask = cv2.imread(latest_obs.seg_mask_path, cv2.IMREAD_GRAYSCALE)
        if loaded_mask is not None and loaded_mask.size > 0:
            seg_mask = loaded_mask
            valid_mask_adv = build_valid_sky_mask_from_config(loaded_mask.shape[:2], config.camera)

    # Check historical measured skill per horizon so horizons failing to beat persistence are honestly flagged
    hist_ver = compute_verification_summary(db)
    hist_by_horizon = hist_ver.get("by_horizon", {}) if isinstance(hist_ver, dict) else {}

    created_forecasts: list[Forecast] = []
    # Camera-only FOV is physically limited to +5/+15/+20 min; +30/+60 min only when external NWP data is fused
    if online_data is not None:
        horizons = getattr(config.forecast, "external_horizons_minutes", None) or [5, 15, 20, 30, 60]
    else:
        horizons = [h for h in (config.forecast.horizons_minutes or [5, 15, 20]) if h <= 20] or [5, 15, 20]
    for idx, h_min in enumerate(horizons):
        # Baseline 1: Persistence
        pers_cov = persistence_baseline_cover(cov_curr)

        # Baseline 2: Optical-Flow pixel mask advection + probabilistic ensemble
        adv_res = (
            advect_cloud_mask_optical_flow(
                seg_mask=seg_mask,
                dx_px_s=dx_px_s,
                dy_px_s=dy_px_s,
                horizon_minutes=float(h_min),
                orig_width=latest_obs.width or None,
                orig_height=latest_obs.height or None,
                valid_mask=valid_mask_adv,
                resolution=int(getattr(config.forecast, "advection_resolution", 96)),
                ensemble_members=int(getattr(config.forecast, "ensemble_members", 5)),
                motion_consistency=float(latest_obs.motion_consistency or 0.5),
            )
            if seg_mask is not None
            else {"available": False}
        )
        of_cov = adv_res.get("advected_cover_pct") if adv_res.get("available") else None
        fov_exit = float(adv_res.get("fov_exit_fraction", 0.0)) if adv_res.get("available") else 0.0

        # Rule-based damped linear projection
        damp = {5: 4.5, 15: 10.5, 20: 13.0, 30: 15.0, 60: 20.0}.get(h_min, min(22.0, h_min * 0.45))
        rule_proj_cov = float(np.clip(cov_curr + rate_per_min * damp, 0.0, 100.0))

        # Blend optical-flow mask advection into local camera forecast when motion is consistent
        if of_cov is not None and float(latest_obs.motion_consistency or 0.0) >= 0.40:
            of_w = float(np.clip(float(latest_obs.motion_consistency) * 0.55 * (1.0 - 0.5 * fov_exit), 0.15, 0.50))
            rule_proj_cov = float(np.clip((1.0 - of_w) * rule_proj_cov + of_w * float(of_cov), 0.0, 100.0))

        # Blend with TemporalSkyConvGRU if available — keep short horizons anchored to local physics/advection
        # so an uncalibrated or synthetic neural checkpoint cannot drag a steady +5m sky off by 20%+.
        if nn_horizon_preds is not None and idx < nn_horizon_preds.shape[0]:
            nn_cov = float(1.0 / (1.0 + np.exp(-nn_horizon_preds[idx, 0])) * 100.0)
            nn_w = {5: 0.12, 15: 0.22, 20: 0.28, 30: 0.32, 60: 0.35}.get(h_min, 0.25)
            h_stat_pre = hist_by_horizon.get(str(h_min), {})
            prev_skill = h_stat_pre.get("skill_vs_persistence")
            if h_stat_pre.get("count", 0) >= 15 and (prev_skill is None or float(prev_skill) < 0.0):
                nn_w *= 0.35
            proj_cov = float(np.clip((1.0 - nn_w) * rule_proj_cov + nn_w * nn_cov, 0.0, 100.0))
            nn_probs = 1.0 / (1.0 + np.exp(-np.clip(nn_horizon_preds[idx, 1:], -12.0, 12.0)))
        else:
            proj_cov = rule_proj_cov
            nn_probs = None

        # Blend with Online Open-Meteo horizon prediction when Online Hybrid is enabled
        online_pprob = None
        if online_data is not None:
            o_h = online_data.get("horizons", {}).get(h_min) or online_data.get("horizons", {}).get(str(h_min))
            if o_h and o_h.get("cloud_cover_pct") is not None:
                # Camera + optical flow dominate +5m/+15m; coarse hourly NWP weight rises toward +30m/+60m
                w_max = float(config.online_weather.fusion_weight_online)
                horizon_ramp = {5: 0.08, 15: 0.25, 20: 0.38, 30: 0.60, 60: 1.0}.get(
                    h_min, min(1.0, max(0.08, h_min / 60.0))
                )
                w_h = w_max * horizon_ramp
                proj_cov = float(np.clip((1.0 - w_h) * proj_cov + w_h * float(o_h["cloud_cover_pct"]), 0.0, 100.0))
                if o_h.get("precip_probability_pct") is not None:
                    online_pprob = float(o_h["precip_probability_pct"]) / 100.0

        delta_cov = proj_cov - cov_curr
        clearing_prob = float(np.clip(0.5 - delta_cov / 35.0, 0.02, 0.96))
        increasing_prob = float(np.clip(0.5 + delta_cov / 35.0, 0.02, 0.96))
        overcast_prob = float(np.clip((proj_cov - 50.0) / 45.0, 0.02, 0.97))
        darkening_prob = float(
            np.clip(0.15 + max(0.0, -lum_delta) / 35.0 + dark_curr / 100.0 * 0.65, 0.02, 0.95)
        )

        cb_prob = float(cls_probs.get("cumulonimbus_like", 0.0))
        precip_prob = (
            float(np.clip(0.45 * (dark_curr / 60.0) + 0.30 * cb_prob + 0.25 * darkening_prob, 0.01, 0.85))
            if config.forecast.rain_prediction_enabled
            else 0.0
        )
        if online_pprob is not None and config.forecast.rain_prediction_enabled:
            precip_prob = float(np.clip(0.45 * precip_prob + 0.55 * online_pprob, 0.01, 0.96))

        storm_prob = (
            float(np.clip(0.50 * cb_prob + 0.35 * (dark_curr / 55.0) + 0.15 * max(0.0, -lum_delta) / 25.0, 0.01, 0.85))
            if config.forecast.experimental_predictions
            else 0.0
        )
        if online_data is not None and config.forecast.experimental_predictions:
            cur_wx = online_data.get("current", {})
            if (cur_wx.get("pressure_msl_hpa") or 1015) < 1005 and (online_pprob or 0.0) > 0.5:
                storm_prob = float(np.clip(storm_prob + 0.20, 0.01, 0.95))

        if nn_probs is not None:
            clearing_prob = round(0.65 * clearing_prob + 0.35 * float(nn_probs[0]), 3)
            increasing_prob = round(0.65 * increasing_prob + 0.35 * float(nn_probs[1]), 3)
            overcast_prob = round(0.65 * overcast_prob + 0.35 * float(nn_probs[2]), 3)
            darkening_prob = round(0.65 * darkening_prob + 0.35 * float(nn_probs[3]), 3)
            if config.forecast.rain_prediction_enabled:
                precip_prob = round(0.70 * precip_prob + 0.30 * float(nn_probs[4]), 3)

        # Confidence decays gracefully with forecast horizon, FOV transit, sky regime, QC artifacts & skill
        hist_factor = min(1.0, len(recent_obs) / 5.0)
        horizon_decay = max(0.32, 1.0 - (h_min / 95.0))
        online_boost = 0.08 if online_data is not None else 0.0
        sky_regime = str(cov_meta.get("sky_regime", "day"))
        q_flags_curr = set(safe_json_loads(latest_obs.quality_metrics_json, {}).get("flags") or [])
        hw_qc_flags = q_flags_curr & {
            "window_reflection",
            "ae_awb_jump",
            "raindrop_smudge",
            "heavy_jpeg_blocking",
        }
        regime_mult = 0.65 if sky_regime == "night" else (0.85 if sky_regime == "twilight" else 1.0)
        qc_mult = 0.80 if hw_qc_flags else 1.0
        conf = float(
            np.clip(
                latest_obs.quality_score * 0.35
                + cov_meta.get("confidence", 0.7) * 0.25
                + hist_factor * 0.20
                + horizon_decay * 0.20
                + online_boost,
                0.15,
                0.95,
            )
            * horizon_decay
            * regime_mult
            * qc_mult
        )

        # Honest reliability assessment vs persistence, narrow camera FOV, night mode & hardware QC limits
        h_stat = hist_by_horizon.get(str(h_min), {})
        h_count = int(h_stat.get("count", 0) or 0)
        h_skill = h_stat.get("skill_vs_persistence")
        low_reliability = False
        reliability_reasons: list[str] = []
        if sky_regime == "night":
            low_reliability = True
            reliability_reasons.append("night_mode_reduced_optical_visibility")
        if hw_qc_flags:
            reliability_reasons.append("hardware_qc_flags:" + ",".join(sorted(hw_qc_flags)))
        if h_count >= 3 and h_skill is not None and float(h_skill) <= 0.0:
            low_reliability = True
            reliability_reasons.append(f"skill_vs_persistence={h_skill:+.2f}_does_not_beat_persistence")
            conf = min(conf, 0.42)
        if h_min >= 30 and (h_skill is None or float(h_skill) <= 0.0):
            low_reliability = True
            reliability_reasons.append("long_horizon_low_camera_visibility")
        if adv_res.get("available") and fov_exit >= 0.70:
            low_reliability = True
            reliability_reasons.append(f"narrow_fov_cloud_transit_{int(fov_exit * 100)}pct")
            conf = min(conf, 0.48)

        if delta_cov >= 10.0:
            cond = "increasing_cloudiness"
        elif delta_cov <= -10.0:
            cond = "clearing"
        else:
            cond = _classify_cover_condition(proj_cov, dark_curr, storm_prob)

        fc_limitations = list(limitations)
        if config.forecast.experimental_predictions and online_data is None:
            fc_limitations.append("precip_and_storm_signals_are_experimental_visual_indicators_only")
        if low_reliability:
            fc_limitations.append("low_reliability_horizon")

        # Probabilistic interval centered around projected coverage using advection ensemble spread
        ens_band = float(adv_res.get("uncertainty_band_pct") or round(4.0 + h_min * 0.42, 1))
        cover_p10 = round(float(np.clip(proj_cov - ens_band, 0.0, 100.0)), 1)
        cover_p90 = round(float(np.clip(proj_cov + ens_band, 0.0, 100.0)), 1)

        baseline_payload = {
            "persistence_cover_pct": pers_cov,
            "optical_flow_cover_pct": of_cov,
            "cover_p10": cover_p10,
            "cover_p50": round(proj_cov, 1),
            "cover_p90": cover_p90,
            "uncertainty_band_pct": round(ens_band, 1),
            "fov_exit_fraction": round(fov_exit, 3),
            "historical_skill_vs_persistence": h_skill,
            "low_reliability": low_reliability,
            "reliability_reasons": reliability_reasons,
        }

        fc = Forecast(
            generated_at=gen_time,
            observation_id=latest_obs.id,
            horizon_minutes=int(h_min),
            target_time=obs_time + timedelta(minutes=int(h_min)),
            status="available",
            forecast_condition=cond,
            cloud_cover_prediction=round(proj_cov, 2),
            clearing_prob=round(clearing_prob, 3),
            increasing_prob=round(increasing_prob, 3),
            overcast_prob=round(overcast_prob, 3),
            darkening_prob=round(darkening_prob, 3),
            precip_signal_prob=round(precip_prob, 3),
            storm_signal_prob=round(storm_prob, 3),
            confidence=round(conf, 3),
            signals_json=json.dumps(signals),
            reasons_json=json.dumps(reasons),
            limitations_json=json.dumps(fc_limitations),
            baseline_metrics_json=json.dumps(baseline_payload),
            model_version=model_version,
            verified=False,
        )
        db.add(fc)
        created_forecasts.append(fc)

    db.commit()
    for fc in created_forecasts:
        db.refresh(fc)
    return created_forecasts
