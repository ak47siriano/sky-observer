import React, { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { Cloud, CloudRain, Crosshair, Globe, Pause, Play } from "lucide-react";
import { useI18n } from "../i18n/I18nContext";
import { api } from "../services/api";

interface SimpleMapPageProps {
  offlineMode: boolean;
  onToggleMode: (offline: boolean) => void;
}

function kmToBounds(lat: number, lon: number, radiusKm: number): L.LatLngTuple[] {
  const dLat = radiusKm / 111.32;
  const cosLat = Math.max(0.15, Math.cos((lat * Math.PI) / 180));
  const dLon = radiusKm / (111.32 * cosLat);
  return [
    [lat - dLat, lon - dLon],
    [lat + dLat, lon + dLon],
  ];
}

function zoomForRadius(radKm: number): number {
  if (radKm <= 3) return 12;
  if (radKm <= 8) return 11;
  if (radKm <= 20) return 10;
  if (radKm <= 45) return 9;
  return 8;
}

function findNearestRainFrame(frames: any[], targetEpochSec: number | null) {
  if (!frames || frames.length === 0) return null;
  if (targetEpochSec == null) return frames[frames.length - 1];
  let best = frames[0];
  let bestDiff = Math.abs(Number(best.time ?? 0) - targetEpochSec);
  for (let i = 1; i < frames.length; i++) {
    const diff = Math.abs(Number(frames[i].time ?? 0) - targetEpochSec);
    if (diff < bestDiff) {
      best = frames[i];
      bestDiff = diff;
    }
  }
  return best;
}

export const SimpleMapPage: React.FC<SimpleMapPageProps> = ({ offlineMode, onToggleMode }) => {
  const { t } = useI18n();
  const [state, setState] = useState<any>(null);
  const [showClouds, setShowClouds] = useState(true);
  const [showRain, setShowRain] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const [frameIdx, setFrameIdx] = useState(0);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const pinRef = useRef<L.Marker | null>(null);
  const fovPolyRef = useRef<L.Polygon | null>(null);
  const cloudOverlayRef = useRef<L.ImageOverlay | null>(null);
  const regionalOverlayRef = useRef<L.ImageOverlay | null>(null);
  const regionalKeyRef = useRef<string>("");
  const rainLayerRef = useRef<L.TileLayer | null>(null);
  const rainUrlRef = useRef<string>("");

  useEffect(() => {
    const load = () => {
      api
        .getRadarState()
        .then((s) => {
          setState(s);
          const hist = Array.isArray(s?.camera_history) ? s.camera_history : [];
          if (hist.length > 0) setFrameIdx(hist.length - 1);
        })
        .catch(() => {});
    };
    load();
    const id = window.setInterval(load, 20000);
    return () => window.clearInterval(id);
  }, [offlineMode]);

  const history: any[] = Array.isArray(state?.camera_history) ? state.camera_history : [];
  const rainFrames: any[] = Array.isArray(state?.rainviewer?.radar_frames) ? state.rainviewer.radar_frames : [];
  const safeIdx = history.length > 0 ? Math.min(Math.max(0, frameIdx), history.length - 1) : 0;
  const activeObs = history.length > 0 ? history[safeIdx] : state?.observation ?? null;
  const activeTimeSec: number | null = activeObs?.time ?? null;
  const matchedRainFrame = findNearestRainFrame(rainFrames, activeTimeSec);

  useEffect(() => {
    if (!isPlaying || history.length <= 1) return;
    const id = window.setInterval(() => {
      setFrameIdx((prev) => (prev + 1 >= history.length ? 0 : prev + 1));
    }, 550);
    return () => window.clearInterval(id);
  }, [isPlaying, history.length]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current || !state?.location) return;
    const lat = Number(state.location.latitude);
    const lon = Number(state.location.longitude);
    const radKm = Number(state.location.radar_radius_km || 15);
    const map = L.map(containerRef.current, {
      center: [lat, lon],
      zoom: zoomForRadius(radKm),
      zoomControl: true,
    });
    L.tileLayer("/api/radar/tiles/dark_base/{z}/{x}/{y}.png", {
      maxZoom: 18,
      attribution: "Tiles &copy; Esri",
    }).addTo(map);
    L.tileLayer("/api/radar/tiles/dark_ref/{z}/{x}/{y}.png", { maxZoom: 18 }).addTo(map);
    mapRef.current = map;
    window.setTimeout(() => {
      map.invalidateSize({ animate: false });
    }, 120);
  }, [state?.location]);

  useEffect(() => {
    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
      pinRef.current = null;
      fovPolyRef.current = null;
      cloudOverlayRef.current = null;
      regionalOverlayRef.current = null;
      regionalKeyRef.current = "";
      rainLayerRef.current = null;
      rainUrlRef.current = "";
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !state?.location) return;
    const lat = Number(state.location.latitude);
    const lon = Number(state.location.longitude);
    const radKm = Number(state.location.radar_radius_km || 15);
    const label = state.location.location_description || t("simple.map.cameraPos");

    if (!pinRef.current) {
      const icon = L.divIcon({
        className: "",
        html: `<div class="radar-pin-wrap"><div class="radar-pin-pulse"></div><div class="radar-pin-core"></div></div>`,
        iconSize: [24, 24],
        iconAnchor: [12, 12],
      });
      pinRef.current = L.marker([lat, lon], { icon, zIndexOffset: 600 }).addTo(map);
    } else {
      pinRef.current.setLatLng([lat, lon]);
    }
    pinRef.current.bindTooltip(label, { direction: "top", offset: [0, -10] });

    // Wide regional cloud cover (when Online + Clouds ON)
    if (!offlineMode && showClouds && Array.isArray(state.regional_cloud_bounds) && state.regional_cloud_bounds.length === 2) {
      const rKey = `${lat.toFixed(3)},${lon.toFixed(3)},${state?.online_weather?.fetched_at || ""}`;
      const rUrl = `/api/radar/regional-clouds.png?lat=${lat.toFixed(3)}&lon=${lon.toFixed(3)}&k=${encodeURIComponent(rKey)}`;
      const rBounds = L.latLngBounds(state.regional_cloud_bounds as L.LatLngTuple[]);
      if (!regionalOverlayRef.current) {
        regionalOverlayRef.current = L.imageOverlay(rUrl, rBounds, {
          opacity: 0.78,
          interactive: false,
        }).addTo(map);
        regionalKeyRef.current = rKey;
      } else if (regionalKeyRef.current !== rKey) {
        regionalOverlayRef.current.setUrl(rUrl);
        regionalOverlayRef.current.setBounds(rBounds);
        regionalKeyRef.current = rKey;
      }
    } else if (regionalOverlayRef.current) {
      map.removeLayer(regionalOverlayRef.current);
      regionalOverlayRef.current = null;
      regionalKeyRef.current = "";
    }

    // Subtle camera field-of-view cone
    if (fovPolyRef.current) {
      map.removeLayer(fovPolyRef.current);
      fovPolyRef.current = null;
    }
    if (showClouds && Array.isArray(state.camera_projection?.fov_polygon) && state.camera_projection.fov_polygon.length >= 3) {
      fovPolyRef.current = L.polygon(state.camera_projection.fov_polygon, {
        color: "#38BDF8",
        weight: 1.5,
        dashArray: "4 4",
        fillColor: "#38BDF8",
        fillOpacity: 0.05,
        interactive: false,
      }).addTo(map);
    }

    // Camera projected cloud overlay (matches backend /api/radar/cloud-overlay.png)
    if (showClouds && activeObs) {
      const rawBounds = (state.camera_projection?.overlay_bounds || kmToBounds(lat, lon, radKm)) as L.LatLngTuple[];
      const bounds = L.latLngBounds(rawBounds);
      const obsQuery = activeObs.id ? `&obs_id=${activeObs.id}` : `&t=${encodeURIComponent(activeObs.captured_at || "")}`;
      const url = `/api/radar/cloud-overlay.png?size=512&el=${state.location.orientation_elevation_deg ?? 65}&az=${state.location.orientation_azimuth_deg ?? 0}&roll=${state.location.orientation_roll_deg ?? 0}&tc=${state.location.tilt_correction_enabled ? 1 : 0}${obsQuery}`;
      if (!cloudOverlayRef.current) {
        cloudOverlayRef.current = L.imageOverlay(url, bounds, {
          opacity: 0.85,
          interactive: false,
        }).addTo(map);
      } else {
        cloudOverlayRef.current.setUrl(url);
        cloudOverlayRef.current.setBounds(bounds);
      }
    } else if (cloudOverlayRef.current) {
      map.removeLayer(cloudOverlayRef.current);
      cloudOverlayRef.current = null;
    }

    // RainViewer precipitation radar (uses frame.tile_url directly)
    const targetRainUrl = !offlineMode && showRain ? matchedRainFrame?.tile_url || "" : "";
    if (!targetRainUrl) {
      if (rainLayerRef.current) {
        map.removeLayer(rainLayerRef.current);
        rainLayerRef.current = null;
      }
      rainUrlRef.current = "";
    } else if (rainUrlRef.current !== targetRainUrl) {
      if (rainLayerRef.current) {
        map.removeLayer(rainLayerRef.current);
      }
      rainLayerRef.current = L.tileLayer(targetRainUrl, {
        maxNativeZoom: 7,
        maxZoom: 18,
        opacity: 0.68,
        zIndex: 400,
        attribution: "Weather data by RainViewer",
      }).addTo(map);
      rainUrlRef.current = targetRainUrl;
    }
  }, [state, activeObs, matchedRainFrame?.tile_url, showClouds, showRain, offlineMode, t]);

  const handleCenter = () => {
    if (!mapRef.current || !state?.location) return;
    const radKm = Number(state.location.radar_radius_km || 15);
    mapRef.current.setView([Number(state.location.latitude), Number(state.location.longitude)], zoomForRadius(radKm), {
      animate: true,
    });
  };

  const cov = activeObs ? Math.round(Number(activeObs.cloud_coverage_pct || 0)) : null;

  return (
    <div className="max-w-[1440px] mx-auto space-y-4 py-1">
      {/* Simple Top Bar */}
      <div className="rounded-2xl border border-white/[0.1] bg-[#151A22] p-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-bold text-white">{t("simple.map.pageTitle")}</h2>
          <p className="text-xs sm:text-sm text-[#9AA1AD]">
            {state?.location?.location_description || t("simple.map.cameraPos")}
            {cov != null && ` · ${cov}% ${t("simple.map.cloudsWord")}`}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setShowClouds((v) => !v)}
            className={`inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs sm:text-sm font-semibold transition ${
              showClouds
                ? "bg-sky-400 text-[#07131D]"
                : "bg-[#10141B] text-[#9AA1AD] border border-white/10"
            }`}
          >
            <Cloud className="w-4 h-4" />
            {t("simple.map.toggleClouds")}
          </button>

          {!offlineMode ? (
            <button
              type="button"
              onClick={() => setShowRain((v) => !v)}
              className={`inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs sm:text-sm font-semibold transition ${
                showRain
                  ? "bg-emerald-400 text-[#062417]"
                  : "bg-[#10141B] text-[#9AA1AD] border border-white/10"
              }`}
            >
              <CloudRain className="w-4 h-4" />
              {t("simple.map.toggleRain")}
            </button>
          ) : (
            <button
              type="button"
              onClick={() => onToggleMode(false)}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs sm:text-sm font-semibold bg-[#10141B] hover:bg-[#1B2230] text-sky-300 border border-sky-500/30 transition"
            >
              <Globe className="w-4 h-4" />
              {t("simple.map.enableRainOnline")}
            </button>
          )}

          <button
            type="button"
            onClick={handleCenter}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs sm:text-sm font-medium bg-[#10141B] hover:bg-[#1B2230] text-white border border-white/10 transition"
          >
            <Crosshair className="w-4 h-4 text-sky-300" />
            {t("simple.map.centerBtn")}
          </button>
        </div>
      </div>

      {/* Big Clean Map Canvas */}
      <div className="relative rounded-2xl overflow-hidden border border-white/[0.1] bg-[#0B0E14] h-[65vh] min-h-[420px] shadow-xl">
        <div ref={containerRef} className="w-full h-full z-10" />

        {history.length > 1 && (
          <div className="absolute bottom-4 left-4 right-4 z-[500] rounded-xl bg-[#0D1117]/85 backdrop-blur-md border border-white/15 px-4 py-2.5 flex items-center gap-3">
            <button
              type="button"
              onClick={() => setIsPlaying((p) => !p)}
              className="w-9 h-9 rounded-full bg-sky-400 text-[#07131D] inline-flex items-center justify-center shrink-0"
              aria-label={isPlaying ? t("radar.pause") : t("radar.play")}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
            </button>
            <input
              type="range"
              min={0}
              max={history.length - 1}
              value={safeIdx}
              onChange={(e) => {
                setIsPlaying(false);
                setFrameIdx(Number(e.target.value));
              }}
              className="flex-1 accent-sky-400"
            />
            <span className="text-xs font-mono text-[#CBD5E1] shrink-0">
              {activeObs?.captured_at ? new Date(activeObs.captured_at).toLocaleTimeString() : ""}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
