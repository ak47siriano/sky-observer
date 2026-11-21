"""Local Weather Nowcasting Engine combining transparent rule-based signals and TemporalSkyConvGRU."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, UTC
from typing import Any

import numpy as np
import torch
from sqlalchemy.orm import Session

from app.ai.registry import get_model_cache
from app.config import AppConfig
from app.models_db import Forecast, Observation, safe_json_loads
from app.services.weather_online import get_online_weather_service


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
            f"Consistent cloud advection toward {latest_obs.motion_direction_cardinal} ({latest_obs.motion_speed_px_s:.1f} px/s)."
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

    created_forecasts: list[Forecast] = []
    # Camera-only FOV is physically limited to +5/+15/+20 min; +30/+60 min only when external NWP data is fused
    if online_data is not None:
        horizons = getattr(config.forecast, "external_horizons_minutes", None) or [5, 15, 20, 30, 60]
    else:
        horizons = [h for h in (config.forecast.horizons_minutes or [5, 15, 20]) if h <= 20] or [5, 15, 20]
    for idx, h_min in enumerate(horizons):
        # Rule-based damped linear projection
        damp = {5: 4.5, 15: 10.5, 20: 13.0, 30: 15.0, 60: 20.0}.get(h_min, min(22.0, h_min * 0.45))
        rule_proj_cov = float(np.clip(cov_curr + rate_per_min * damp, 0.0, 100.0))

        # Blend with TemporalSkyConvGRU if available
        if nn_horizon_preds is not None and idx < nn_horizon_preds.shape[0]:
            nn_cov = float(1.0 / (1.0 + np.exp(-nn_horizon_preds[idx, 0])) * 100.0)
            proj_cov = float(np.clip(0.60 * rule_proj_cov + 0.40 * nn_cov, 0.0, 100.0))
            nn_probs = 1.0 / (1.0 + np.exp(-np.clip(nn_horizon_preds[idx, 1:], -12.0, 12.0)))
        else:
            proj_cov = rule_proj_cov
            nn_probs = None

        # Blend with Online Open-Meteo horizon prediction when Online Hybrid is enabled
        online_pprob = None
        if online_data is not None:
            o_h = online_data.get("horizons", {}).get(h_min) or online_data.get("horizons", {}).get(str(h_min))
            if o_h and o_h.get("cloud_cover_pct") is not None:
                # Camera dominates at +5m, regional NWP weight increases smoothly toward +60m
                w_max = float(config.online_weather.fusion_weight_online)
                w_h = w_max * min(1.0, 0.35 + (h_min / 60.0) * 0.65)
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

        # Confidence decays gracefully with forecast horizon and low quality/history
        hist_factor = min(1.0, len(recent_obs) / 5.0)
        horizon_decay = max(0.32, 1.0 - (h_min / 95.0))
        online_boost = 0.08 if online_data is not None else 0.0
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
        )

        if delta_cov >= 10.0:
            cond = "increasing_cloudiness"
        elif delta_cov <= -10.0:
            cond = "clearing"
        else:
            cond = _classify_cover_condition(proj_cov, dark_curr, storm_prob)

        fc_limitations = list(limitations)
        if config.forecast.experimental_predictions and online_data is None:
            fc_limitations.append("precip_and_storm_signals_are_experimental_visual_indicators_only")

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
            model_version=model_version,
            verified=False,
        )
        db.add(fc)
        created_forecasts.append(fc)

    db.commit()
    for fc in created_forecasts:
        db.refresh(fc)
    return created_forecasts
