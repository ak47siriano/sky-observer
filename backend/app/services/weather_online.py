"""Free, zero-registration Online Weather & Radar Service (Open-Meteo + RainViewer) with strict offline lock."""

from __future__ import annotations

import logging
import time
from datetime import datetime, UTC
from threading import RLock
from typing import Any

import httpx

from app.config import AppConfig

logger = logging.getLogger(__name__)

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
RAINVIEWER_MAPS_URL = "https://api.rainviewer.com/public/weather-maps.json"
BIGDATACLOUD_REVERSE_URL = "https://api.bigdatacloud.net/data/reverse-geocode-client"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
_REVERSE_HEADERS = {"User-Agent": "SkyObserver/1.0 (local cloud telemetry)"}


def _build_place(city: str, region: str, country: str) -> dict[str, str] | None:
    c = city.strip()
    if not c:
        return None
    r = region.strip()
    co = country.strip()
    parts = [p for p in [c, r or co] if p]
    return {
        "city": c,
        "region": r,
        "country": co,
        "label": ", ".join(dict.fromkeys(parts)),
    }


def _estimate_dewpoint_and_lcl(temp_c: Any, rh_pct: Any) -> tuple[float | None, float | None]:
    """Compute Magnus-Tetens dewpoint (°C) and Espy Lifting Condensation Level cloud base (m)."""
    if temp_c is None or rh_pct is None:
        return None, None
    try:
        t = float(temp_c)
        rh = max(1.0, min(100.0, float(rh_pct)))
        import math

        gamma = (17.625 * t) / (243.04 + t) + math.log(rh / 100.0)
        t_dew = (243.04 * gamma) / (17.625 - gamma)
        lcl_m = max(300.0, min(6000.0, 125.0 * max(0.0, t - t_dew)))
        return round(t_dew, 1), round(lcl_m, 0)
    except Exception:
        return None, None


class OnlineWeatherService:
    """
    Fetches live weather & regional cloud grids from Open-Meteo and radar tiles from RainViewer.
    Zero registration or API keys required.
    Never opens outbound network connections when config.server.offline_mode is True.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._weather_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._grid_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._radar_cache: tuple[float, dict[str, Any]] | None = None
        self._reverse_cache: dict[str, tuple[float, dict[str, str]]] = {}

    def clear_cache(self) -> None:
        with self._lock:
            self._weather_cache.clear()
            self._grid_cache.clear()
            self._radar_cache = None
            self._reverse_cache.clear()

    def reverse_geocode(self, lat: float, lon: float, config: Any, lang: str = "it") -> dict[str, str] | None:
        """Resolve city/region/country for coordinates using free zero-registration APIs."""
        if config.server.offline_mode:
            return None
        key = f"{lat:.3f}:{lon:.3f}:{lang}"
        now = datetime.now(UTC).timestamp()
        with self._lock:
            cached = self._reverse_cache.get(key)
            if cached and (now - cached[0]) < 86400.0:
                return cached[1]

        timeout = float(getattr(config.server, "online_weather_timeout_s", 6.0) or 6.0)
        place: dict[str, str] | None = None
        providers = (
            (
                BIGDATACLOUD_REVERSE_URL,
                {"latitude": round(lat, 5), "longitude": round(lon, 5), "localityLanguage": lang},
                lambda d: _build_place(
                    str(d.get("city") or d.get("locality") or ""),
                    str(d.get("principalSubdivision") or ""),
                    str(d.get("countryName") or ""),
                ),
            ),
            (
                NOMINATIM_REVERSE_URL,
                {
                    "format": "jsonv2",
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "zoom": 12,
                    "addressdetails": 1,
                    "accept-language": lang,
                },
                lambda d: _build_place(
                    str(
                        (d.get("address") or {}).get("city")
                        or (d.get("address") or {}).get("town")
                        or (d.get("address") or {}).get("village")
                        or (d.get("address") or {}).get("municipality")
                        or d.get("name")
                        or ""
                    ),
                    str((d.get("address") or {}).get("state") or (d.get("address") or {}).get("county") or ""),
                    str((d.get("address") or {}).get("country") or ""),
                ),
            ),
        )
        try:
            with httpx.Client(timeout=timeout, headers=_REVERSE_HEADERS) as client:
                for url, params, extractor in providers:
                    try:
                        r = client.get(url, params=params)
                        r.raise_for_status()
                        place = extractor(r.json())
                        if place is not None:
                            break
                    except Exception:
                        continue
        except Exception:
            place = None

        if place is not None:
            with self._lock:
                self._reverse_cache[key] = (now, place)
        return place

    def get_weather(self, lat: float, lon: float, config: AppConfig) -> dict[str, Any]:
        if config.server.offline_mode:
            return {"offline_mode": True, "available": False, "reason": "Offline mode active"}

        key = f"{lat:.3f},{lon:.3f}"
        now = time.time()
        with self._lock:
            cached = self._weather_cache.get(key)
            if cached and (now - cached[0]) < config.online_weather.cache_ttl_seconds:
                return cached[1]

        try:
            params = {
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "current": ",".join([
                    "temperature_2m",
                    "relative_humidity_2m",
                    "apparent_temperature",
                    "pressure_msl",
                    "cloud_cover",
                    "cloud_cover_low",
                    "cloud_cover_mid",
                    "cloud_cover_high",
                    "precipitation",
                    "rain",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "weather_code",
                ]),
                "minutely_15": ",".join([
                    "cloud_cover",
                    "precipitation",
                    "precipitation_probability",
                    "visibility",
                ]),
                "forecast_minutely_15": 8,
                "timezone": "UTC",
            }
            with httpx.Client(timeout=8.0) as client:
                resp = client.get(OPEN_METEO_FORECAST_URL, params=params)
                resp.raise_for_status()
                raw = resp.json()

            cur = raw.get("current", {})
            m15 = raw.get("minutely_15", {})
            m_cov = m15.get("cloud_cover", []) or []
            m_pprob = m15.get("precipitation_probability", []) or []
            m_precip = m15.get("precipitation", []) or []
            m_times = m15.get("time", []) or []

            # Interpolate 5m, 15m, 30m, 60m horizons from current + 15-minutely steps
            c0 = float(cur.get("cloud_cover", m_cov[0] if m_cov else 0.0) or 0.0)
            c15 = float(m_cov[1] if len(m_cov) > 1 else (m_cov[0] if m_cov else c0))
            c30 = float(m_cov[2] if len(m_cov) > 2 else c15)
            c60 = float(m_cov[4] if len(m_cov) > 4 else (m_cov[-1] if m_cov else c30))
            c5 = round(c0 * 0.67 + c15 * 0.33, 1)

            p15 = float(m_pprob[1] if len(m_pprob) > 1 else (m_pprob[0] if m_pprob else 0.0) or 0.0)
            p30 = float(m_pprob[2] if len(m_pprob) > 2 else p15 or 0.0)
            p60 = float(m_pprob[4] if len(m_pprob) > 4 else (m_pprob[-1] if m_pprob else p30) or 0.0)

            place: dict[str, str] | None = None
            try:
                place = self.reverse_geocode(lat, lon, config)
            except Exception:
                place = None
            dew_c, lcl_m = _estimate_dewpoint_and_lcl(cur.get("temperature_2m"), cur.get("relative_humidity_2m"))
            payload = {
                "offline_mode": False,
                "available": True,
                "provider": "Open-Meteo (No-Key Free NWP)",
                "latitude": raw.get("latitude", lat),
                "longitude": raw.get("longitude", lon),
                "elevation_m": raw.get("elevation"),
                "city": place["city"] if place else None,
                "region": place["region"] if place else None,
                "country": place["country"] if place else None,
                "location_name": place["label"] if place else None,
                "fetched_at": datetime.now(UTC).isoformat(),
                "current": {
                    "temperature_c": cur.get("temperature_2m"),
                    "apparent_temperature_c": cur.get("apparent_temperature"),
                    "relative_humidity_pct": cur.get("relative_humidity_2m"),
                    "dewpoint_c": dew_c,
                    "estimated_cloud_base_m": lcl_m,
                    "pressure_msl_hpa": cur.get("pressure_msl"),
                    "cloud_cover_pct": cur.get("cloud_cover"),
                    "cloud_cover_low_pct": cur.get("cloud_cover_low"),
                    "cloud_cover_mid_pct": cur.get("cloud_cover_mid"),
                    "cloud_cover_high_pct": cur.get("cloud_cover_high"),
                    "precipitation_mm": cur.get("precipitation"),
                    "rain_mm": cur.get("rain"),
                    "wind_speed_kmh": cur.get("wind_speed_10m"),
                    "wind_direction_deg": cur.get("wind_direction_10m"),
                    "weather_code": cur.get("weather_code"),
                },
                "horizons": {
                    5: {"cloud_cover_pct": c5, "precip_probability_pct": round(p15 * 0.6, 1)},
                    15: {"cloud_cover_pct": round(c15, 1), "precip_probability_pct": round(p15, 1)},
                    30: {"cloud_cover_pct": round(c30, 1), "precip_probability_pct": round(p30, 1)},
                    60: {"cloud_cover_pct": round(c60, 1), "precip_probability_pct": round(p60, 1)},
                },
                "minutely_15_series": [
                    {
                        "time": m_times[i] if i < len(m_times) else None,
                        "cloud_cover_pct": m_cov[i] if i < len(m_cov) else None,
                        "precip_probability_pct": m_pprob[i] if i < len(m_pprob) else None,
                        "precipitation_mm": m_precip[i] if i < len(m_precip) else None,
                    }
                    for i in range(min(8, max(len(m_times), len(m_cov))))
                ],
            }
            with self._lock:
                self._weather_cache[key] = (now, payload)
            return payload
        except Exception as exc:
            logger.warning("Online weather request failed: %s", exc)
            return {
                "offline_mode": False,
                "available": False,
                "provider": "Open-Meteo",
                "error": "Online weather provider temporarily unavailable",
            }

    def get_regional_cloud_grid(
        self, lat: float, lon: float, config: AppConfig, span_deg: float = 1.4, grid_n: int = 5
    ) -> list[dict[str, Any]]:
        """Query a 5x5 regional Open-Meteo cloud grid in one batch HTTP call for map rendering."""
        if config.server.offline_mode:
            return []

        key = f"grid:{lat:.2f},{lon:.2f},{span_deg:.2f},{grid_n}"
        now = time.time()
        with self._lock:
            cached = self._grid_cache.get(key)
            if cached and (now - cached[0]) < config.online_weather.cache_ttl_seconds:
                return cached[1]

        half = span_deg / 2.0
        lats: list[float] = []
        lons: list[float] = []
        for r in range(grid_n):
            la = max(-85.0, min(85.0, lat - half + (r / max(1, grid_n - 1)) * span_deg))
            for c in range(grid_n):
                lo = ((lon - half + (c / max(1, grid_n - 1)) * span_deg + 180.0) % 360.0) - 180.0
                lats.append(round(la, 3))
                lons.append(round(lo, 3))

        try:
            params = {
                "latitude": ",".join(str(x) for x in lats),
                "longitude": ",".join(str(x) for x in lons),
                "current": "cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,precipitation",
                "timezone": "UTC",
            }
            with httpx.Client(timeout=9.0) as client:
                resp = client.get(OPEN_METEO_FORECAST_URL, params=params)
                resp.raise_for_status()
                raw = resp.json()

            rows = raw if isinstance(raw, list) else [raw]
            points: list[dict[str, Any]] = []
            for idx, item in enumerate(rows):
                cur = item.get("current", {})
                points.append({
                    "latitude": item.get("latitude", lats[idx] if idx < len(lats) else lat),
                    "longitude": item.get("longitude", lons[idx] if idx < len(lons) else lon),
                    "cloud_cover_pct": float(cur.get("cloud_cover") or 0.0),
                    "cloud_low_pct": float(cur.get("cloud_cover_low") or 0.0),
                    "cloud_mid_pct": float(cur.get("cloud_cover_mid") or 0.0),
                    "cloud_high_pct": float(cur.get("cloud_cover_high") or 0.0),
                    "precipitation_mm": float(cur.get("precipitation") or 0.0),
                })

            with self._lock:
                self._grid_cache[key] = (now, points)
            return points
        except Exception:
            return []

    def get_radar_layers(self, config: AppConfig) -> dict[str, Any] | None:
        """
        Fetch precipitation radar tile frames (RainViewer free tier: past radar only, max zoom 7,
        Universal Blue color scheme, dynamic hash paths, or optional self-hosted LibreWXR tile URL).
        """
        if config.server.offline_mode:
            return None
        provider_mode = getattr(config.online_weather, "radar_provider", "rainviewer")
        if provider_mode == "disabled" or not config.online_weather.rainviewer_enabled:
            return None

        if provider_mode == "librewxr":
            tile_tpl = (getattr(config.online_weather, "librewxr_tile_url", "") or "").strip()
            if not tile_tpl:
                return None
            return {
                "provider": "LibreWXR (Self-Hosted)",
                "attribution": "Weather radar via LibreWXR",
                "attribution_url": "https://github.com/librewxr/librewxr",
                "max_zoom": 10,
                "generated": int(time.time()),
                "radar_frames": [
                    {
                        "time": int(time.time()),
                        "kind": "radar_past",
                        "tile_url": tile_tpl,
                    }
                ],
                "satellite_frames": [],
            }

        now = time.time()
        with self._lock:
            if self._radar_cache and (now - self._radar_cache[0]) < 180.0:
                return self._radar_cache[1]

        try:
            with httpx.Client(timeout=7.0, headers=_REVERSE_HEADERS) as client:
                resp = client.get(RAINVIEWER_MAPS_URL)
                resp.raise_for_status()
                data = resp.json()

            host = data.get("host", "https://tilecache.rainviewer.com")
            # RainViewer free tier provides past radar frames only (2h window, max zoom 7,
            # Universal Blue scheme = 2, dynamic hash paths); nowcast & satellite IR are discontinued.
            past = data.get("radar", {}).get("past", []) or []

            frames = []
            for item in past[-8:]:
                path = item.get("path", "")
                if path:
                    frames.append({
                        "time": item.get("time"),
                        "kind": "radar_past",
                        "tile_url": f"{host}{path}/256/{{z}}/{{x}}/{{y}}/2/1_1.png",
                    })

            out = {
                "provider": "RainViewer",
                "attribution": "Weather data by RainViewer",
                "attribution_url": "https://www.rainviewer.com/api.html",
                "max_zoom": 7,
                "generated": data.get("generated"),
                "radar_frames": frames,
                "satellite_frames": [],
            }
            with self._lock:
                self._radar_cache = (now, out)
            return out
        except Exception:
            return None

    def geocode_search(self, query: str, lang: str, config: AppConfig) -> list[dict[str, Any]]:
        if config.server.offline_mode or len(query.strip()) < 2:
            return []
        try:
            params = {
                "name": query.strip(),
                "count": 6,
                "language": "it" if lang == "it" else "en",
                "format": "json",
            }
            with httpx.Client(timeout=7.0) as client:
                resp = client.get(OPEN_METEO_GEOCODE_URL, params=params)
                resp.raise_for_status()
                results = resp.json().get("results", []) or []
            return [
                {
                    "name": r.get("name"),
                    "admin1": r.get("admin1"),
                    "country": r.get("country"),
                    "latitude": r.get("latitude"),
                    "longitude": r.get("longitude"),
                }
                for r in results
            ]
        except Exception:
            return []


_online_weather_service = OnlineWeatherService()


def get_online_weather_service() -> OnlineWeatherService:
    return _online_weather_service
