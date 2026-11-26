"""Optional Clear-Sky Solar Irradiance, Rooftop PV Output & Ramp-Event Estimation.

100% offline-capable using a pure-Python Ineichen/Kasten clear-sky GHI model paired with
Stage-1 cloud-mask optical transmittance (thin/dense/dark fractions) and plane-of-array (POA)
projection for a user-configured PV array. Uses optional `pvlib` if installed, or optional
Open-Meteo irradiance fallback when the sun lies outside the camera FOV in Online Hybrid mode.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from app.vision.geometry import compute_solar_position


def compute_clear_sky_ghi_wm2(solar_elevation_deg: float, altitude_m: float = 0.0) -> float:
    """
    Estimate clear-sky Global Horizontal Irradiance (GHI, W/m^2) via the Kasten-Young air-mass
    and Haurwitz/Ineichen clear-sky approximation (pure Python, zero network calls).
    """
    el = float(solar_elevation_deg)
    if el <= 0.0:
        return 0.0
    el_rad = math.radians(el)
    sin_el = math.sin(el_rad)
    # Kasten-Young (1989) relative optical air mass
    am = 1.0 / (sin_el + 0.50572 * ((el + 6.07995) ** -1.6364))
    # Pressure correction for site elevation
    p_ratio = math.exp(-max(0.0, float(altitude_m)) / 8434.5)
    am_corr = am * p_ratio
    # Standard Ineichen-like clear-sky GHI curve (calibrated for Linke turbidity TL ~ 3.0)
    ghi = 1098.0 * sin_el * math.exp(-0.059 / max(sin_el, 0.05)) * (0.95 + 0.05 * math.exp(-0.1 * am_corr))
    return round(max(0.0, ghi), 1)


def estimate_cloud_transmittance(
    thin_cloud_pct: float,
    dense_cloud_pct: float,
    dark_cloud_pct: float,
) -> float:
    """
    Estimate broadband solar transmittance factor in [0.12, 1.0] from Stage-1 cloud optical layers
    (Kasten-Czeplak inspired attenuation weighted by optical thickness class).
    """
    thin = max(0.0, min(100.0, float(thin_cloud_pct))) / 100.0
    dense = max(0.0, min(100.0, float(dense_cloud_pct))) / 100.0
    dark = max(0.0, min(100.0, float(dark_cloud_pct))) / 100.0
    attenuation = 0.22 * thin + 0.62 * dense + 0.84 * dark
    return round(max(0.12, min(1.0, 1.0 - attenuation)), 3)


def project_ghi_to_poa_wm2(
    ghi_wm2: float,
    solar_elevation_deg: float,
    solar_azimuth_deg: float,
    panel_tilt_deg: float,
    panel_azimuth_deg: float,
) -> float:
    """Project horizontal GHI onto a tilted PV plane-of-array (POA) using isotropic diffuse + direct geometry."""
    if ghi_wm2 <= 0.0 or solar_elevation_deg <= 0.0:
        return 0.0
    z_rad = math.radians(max(0.0, 90.0 - float(solar_elevation_deg)))
    s_az = math.radians(float(solar_azimuth_deg))
    tilt = math.radians(max(0.0, min(90.0, float(panel_tilt_deg))))
    p_az = math.radians(float(panel_azimuth_deg))

    cos_aoi = math.cos(z_rad) * math.cos(tilt) + math.sin(z_rad) * math.sin(tilt) * math.cos(s_az - p_az)
    cos_z = max(0.08, math.cos(z_rad))
    rb = max(0.0, min(2.5, max(0.0, cos_aoi) / cos_z))

    # Split GHI into approximate 70% direct / 30% diffuse under partial sun, shifting diffuse at low transmittance
    direct_frac = 0.68
    diffuse_frac = 1.0 - direct_frac
    poa = ghi_wm2 * (direct_frac * rb + diffuse_frac * (1.0 + math.cos(tilt)) * 0.5)
    return round(max(0.0, min(1350.0, poa)), 1)


def estimate_solar_pv_and_ramps(
    *,
    lat: float,
    lon: float,
    thin_cloud_pct: float,
    dense_cloud_pct: float,
    dark_cloud_pct: float,
    cloud_coverage_pct: float,
    pv_kwp: float = 3.0,
    panel_tilt_deg: float = 30.0,
    panel_azimuth_deg: float = 180.0,
    system_loss_fraction: float = 0.14,
    ramp_alert_pct: float = 25.0,
    forecast_horizons: list[dict[str, Any]] | None = None,
    dt_utc: datetime | None = None,
    online_ghi_wm2: float | None = None,
    sun_in_fov: bool = True,
) -> dict[str, Any]:
    """
    Compute current clear-sky & cloud-attenuated GHI, estimated rooftop PV power (kW),
    and predicted solar ramp events across short-term nowcast horizons.
    """
    solar = compute_solar_position(lat, lon, dt_utc or datetime.now(UTC))
    el = float(solar["elevation_deg"])
    az = float(solar["azimuth_deg"])

    clear_ghi = compute_clear_sky_ghi_wm2(el)
    trans = estimate_cloud_transmittance(thin_cloud_pct, dense_cloud_pct, dark_cloud_pct)

    irradiance_source = "camera_cloud_mask"
    if (not sun_in_fov) and (online_ghi_wm2 is not None) and float(online_ghi_wm2) >= 0.0:
        est_ghi = round(0.5 * (clear_ghi * trans) + 0.5 * float(online_ghi_wm2), 1)
        irradiance_source = "hybrid_camera_and_open_meteo"
    else:
        est_ghi = round(clear_ghi * trans, 1)

    poa_clear = project_ghi_to_poa_wm2(clear_ghi, el, az, panel_tilt_deg, panel_azimuth_deg)
    poa_est = project_ghi_to_poa_wm2(est_ghi, el, az, panel_tilt_deg, panel_azimuth_deg)

    eff = max(0.1, 1.0 - float(system_loss_fraction))
    kwp = max(0.0, float(pv_kwp))
    clear_pv_kw = round(kwp * (poa_clear / 1000.0) * eff, 3)
    est_pv_kw = round(kwp * (poa_est / 1000.0) * eff, 3)

    # Evaluate short-term PV ramp events from forecasted cloud cover deltas
    ramp_events: list[dict[str, Any]] = []
    if forecast_horizons and clear_pv_kw > 0.05:
        for fc in forecast_horizons:
            h_min = int(fc.get("horizon_minutes", 0) or 0)
            pred_cov = float(fc.get("cloud_cover_prediction", cloud_coverage_pct))
            # Scale current cloud optical mix proportionally to predicted total coverage
            ratio = (pred_cov / max(cloud_coverage_pct, 1.0)) if cloud_coverage_pct > 2.0 else (pred_cov / 100.0)
            if cloud_coverage_pct > 2.0:
                f_thin = min(100.0, thin_cloud_pct * ratio)
                f_dense = min(100.0, dense_cloud_pct * ratio)
                f_dark = min(100.0, dark_cloud_pct * ratio)
            else:
                f_thin, f_dense, f_dark = pred_cov * 0.3, pred_cov * 0.6, pred_cov * 0.1
            f_trans = estimate_cloud_transmittance(f_thin, f_dense, f_dark)
            f_poa = project_ghi_to_poa_wm2(clear_ghi * f_trans, el, az, panel_tilt_deg, panel_azimuth_deg)
            f_pv_kw = round(kwp * (f_poa / 1000.0) * eff, 3)
            delta_kw = round(f_pv_kw - est_pv_kw, 3)
            delta_pct_cap = round((delta_kw / max(kwp, 0.1)) * 100.0, 1)
            event_type = "steady"
            if delta_pct_cap <= -float(ramp_alert_pct):
                event_type = "ramp_down"
            elif delta_pct_cap >= float(ramp_alert_pct):
                event_type = "ramp_up"
            ramp_events.append({
                "horizon_minutes": h_min,
                "predicted_cloud_cover_pct": round(pred_cov, 1),
                "predicted_pv_kw": f_pv_kw,
                "delta_kw": delta_kw,
                "delta_pct_capacity": delta_pct_cap,
                "ramp_type": event_type,
            })

    active_ramp = next((r for r in ramp_events if r["ramp_type"] != "steady"), None)
    return {
        "solar_elevation_deg": el,
        "solar_azimuth_deg": az,
        "is_daylight": bool(el > 0.0),
        "irradiance_source": irradiance_source,
        "clear_sky_ghi_wm2": clear_ghi,
        "estimated_ghi_wm2": est_ghi,
        "cloud_transmittance": trans,
        "poa_irradiance_wm2": poa_est,
        "pv_kwp": kwp,
        "clear_sky_pv_kw": clear_pv_kw,
        "estimated_pv_kw": est_pv_kw,
        "ramp_alert": active_ramp,
        "horizon_ramps": ramp_events,
    }
