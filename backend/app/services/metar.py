"""Optional External METAR Verification (Okta Cloud Cover & Ceilometer Cloud Base Altitude).

Fetches official airport METAR reports from AviationWeather.gov (no API key required) strictly
when offline_mode is False, and provides a deterministic offline raw-METAR parser for testing
and local verification against camera cloud coverage and estimated cloud-base height.
"""

from __future__ import annotations

import re
from datetime import datetime, UTC
from typing import Any

import httpx

from app.config import AppConfig

AVIATIONWEATHER_METAR_URL = "https://aviationweather.gov/api/data/metar"
_HTTP_HEADERS = {"User-Agent": "SkyObserver/1.0 (Local Sky Observatory Verification)"}

# Standard WMO/ICAO okta midpoint percentages for METAR cloud-cover abbreviations
_METAR_OKTA_MAP: dict[str, tuple[int, float]] = {
    "SKC": (0, 0.0),
    "CLR": (0, 0.0),
    "NSC": (0, 0.0),
    "NCD": (0, 0.0),
    "CAVOK": (0, 0.0),
    "FEW": (2, 18.75),   # 1-2 oktas
    "SCT": (4, 43.75),   # 3-4 oktas
    "BKN": (6, 75.0),    # 5-7 oktas
    "OVC": (8, 100.0),   # 8 oktas
    "VV": (8, 100.0),    # Vertical visibility (obscured sky)
}

_LAYER_RE = re.compile(r"\b(FEW|SCT|BKN|OVC|VV)(\d{3}|///)(CB|TCU)?\b")
_CLEAR_RE = re.compile(r"\b(SKC|CLR|NSC|NCD|CAVOK)\b")


def parse_metar_cloud_layers(raw_metar: str) -> dict[str, Any]:
    """Parse cloud okta fraction and lowest ceilometer cloud-base height (m) from a raw METAR string."""
    text = (raw_metar or "").strip().upper()
    layers: list[dict[str, Any]] = []
    max_oktas = 0
    max_cover_pct = 0.0
    lowest_base_m: float | None = None
    has_convective = False

    for match in _LAYER_RE.finditer(text):
        code, h_str, modifier = match.groups()
        oktas, pct = _METAR_OKTA_MAP[code]
        base_m: float | None = None
        if h_str.isdigit():
            # METAR heights are reported in hundreds of feet AGL
            base_m = round(int(h_str) * 100.0 * 0.3048, 1)
            if lowest_base_m is None or base_m < lowest_base_m:
                lowest_base_m = base_m
        if oktas >= max_oktas:
            max_oktas = oktas
            max_cover_pct = pct
        if modifier in {"CB", "TCU"}:
            has_convective = True
        layers.append({
            "code": code,
            "oktas": oktas,
            "cover_pct": pct,
            "base_m": base_m,
            "convective_tag": modifier or None,
        })

    if not layers and _CLEAR_RE.search(text):
        m_clr = _CLEAR_RE.search(text)
        code = m_clr.group(1) if m_clr else "CLR"
        layers.append({"code": code, "oktas": 0, "cover_pct": 0.0, "base_m": None, "convective_tag": None})

    return {
        "raw_metar": raw_metar.strip(),
        "layers": layers,
        "oktas": max_oktas,
        "cloud_cover_pct": round(max_cover_pct, 1),
        "ceilometer_base_m": lowest_base_m,
        "has_convective_tag": has_convective,
        "parsed": bool(layers),
    }


def fetch_nearest_metar(
    lat: float,
    lon: float,
    config: AppConfig,
    icao: str | None = None,
    radius_deg: float = 0.8,
) -> dict[str, Any]:
    """
    Query AviationWeather.gov for the nearest airport METAR (or explicit ICAO station).
    Strictly disabled when config.server.offline_mode is True.
    """
    if config.server.offline_mode:
        return {"available": False, "offline_mode": True, "reason": "Offline mode active"}

    params: dict[str, Any] = {"format": "json"}
    if icao and icao.strip():
        params["ids"] = icao.strip().upper()[:8]
    else:
        params["bbox"] = f"{lat - radius_deg:.3f},{lon - radius_deg:.3f},{lat + radius_deg:.3f},{lon + radius_deg:.3f}"

    try:
        with httpx.Client(timeout=8.0, headers=_HTTP_HEADERS) as client:
            resp = client.get(AVIATIONWEATHER_METAR_URL, params=params)
            resp.raise_for_status()
            rows = resp.json()
        if not isinstance(rows, list) or not rows:
            return {"available": False, "offline_mode": False, "reason": "No nearby METAR station found"}

        # Choose geographically closest station
        def _dist2(r: dict[str, Any]) -> float:
            raw_lat = r.get("lat")
            raw_lon = r.get("lon")
            rlat = float(raw_lat) if raw_lat is not None else lat + 99.0
            rlon = float(raw_lon) if raw_lon is not None else lon + 99.0
            return (rlat - lat) ** 2 + (rlon - lon) ** 2

        best = min(rows, key=_dist2)
        raw_ob = str(best.get("rawOb") or best.get("raw_text") or "")
        parsed = parse_metar_cloud_layers(raw_ob)
        return {
            "available": True,
            "offline_mode": False,
            "station_icao": str(best.get("icaoId") or icao or "").upper(),
            "station_name": str(best.get("name") or ""),
            "station_lat": best.get("lat"),
            "station_lon": best.get("lon"),
            "report_time": best.get("reportTime"),
            "fetched_at": datetime.now(UTC).isoformat(),
            **parsed,
        }
    except Exception:
        return {"available": False, "offline_mode": False, "reason": "METAR provider temporarily unavailable"}


def compare_observation_with_metar(
    camera_cloud_pct: float,
    estimated_base_m: float | None,
    metar_report: dict[str, Any],
) -> dict[str, Any]:
    """Compute objective discrepancies between camera estimates and airport METAR/ceilometer observations."""
    if not metar_report.get("available") and not metar_report.get("parsed"):
        return {"comparable": False, "metar": metar_report}

    m_cov = float(metar_report.get("cloud_cover_pct", 0.0))
    m_base = metar_report.get("ceilometer_base_m")
    cov_err = round(float(camera_cloud_pct) - m_cov, 2)
    base_err = (
        round(float(estimated_base_m) - float(m_base), 1)
        if (estimated_base_m is not None and m_base is not None)
        else None
    )
    return {
        "comparable": True,
        "camera_cloud_pct": round(float(camera_cloud_pct), 2),
        "metar_cloud_pct": m_cov,
        "metar_oktas": int(metar_report.get("oktas", 0)),
        "coverage_error_pct": cov_err,
        "coverage_abs_error_pct": round(abs(cov_err), 2),
        "camera_estimated_base_m": estimated_base_m,
        "metar_ceilometer_base_m": m_base,
        "base_height_error_m": base_err,
        "metar": metar_report,
    }
