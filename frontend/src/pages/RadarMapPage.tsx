import React, { useCallback, useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import {
  ChevronDown,
  ChevronUp,
  Cloud,
  Crosshair,
  Download,
  Droplets,
  Eye,
  Globe,
  LocateFixed,
  Lock,
  MapPin,
  Maximize2,
  Minimize2,
  Pause,
  Play,
  RefreshCw,
  Search,
  Sliders,
} from "lucide-react";
import { CameraTiltCalibrator, type CameraGeometryValues } from "../components/CameraTiltCalibrator";
import { RadarTelemetrySidebar } from "../components/radar/RadarTelemetrySidebar";
import { useI18n } from "../i18n/I18nContext";
import { api } from "../services/api";

interface Props {
  offlineMode: boolean;
  onToggleMode: (offline: boolean) => void;
  onLocationSaved: () => void;
}

function kmToBounds(lat: number, lon: number, radiusKm: number): L.LatLngBoundsExpression {
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

function degToCardinal(deg: number): string {
  const dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  return dirs[Math.floor(((deg + 22.5) % 360) / 45)] || "N";
}

function pointInPolygon(lat: number, lon: number, poly: number[][]): boolean {
  if (!Array.isArray(poly) || poly.length < 3) return false;
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const yi = poly[i][0];
    const xi = poly[i][1];
    const yj = poly[j][0];
    const xj = poly[j][1];
    const intersect = yi > lat !== yj > lat && lon < ((xj - xi) * (lat - yi)) / (yj - yi + 1e-12) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

export const RadarMapPage: React.FC<Props> = ({ offlineMode, onToggleMode, onLocationSaved }) => {
  const { locale, t, tCardinal } = useI18n();
  const [state, setState] = useState<any>(null);
  const [latInput, setLatInput] = useState<string>("");
  const [lonInput, setLonInput] = useState<string>("");
  const [radiusInput, setRadiusInput] = useState<string>("15");
  const [locDesc, setLocDesc] = useState<string>("");
  const [searchQ, setSearchQ] = useState("");
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [showCameraDome, setShowCameraDome] = useState(true);
  const [showRegionalClouds, setShowRegionalClouds] = useState(true);
  const [showRainRadar, setShowRainRadar] = useState(true);
  const [mapStyle, setMapStyle] = useState<"dark" | "osm" | "satellite">("dark");
  const [timelineIdx, setTimelineIdx] = useState<number>(0);
  const [isLive, setIsLive] = useState<boolean>(true);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [mapClickMode, setMapClickMode] = useState<"inspect" | "pin">("inspect");
  const [inspectedPoint, setInspectedPoint] = useState<any | null>(null);
  const [showPip, setShowPip] = useState<boolean>(true);
  const [pipLayer, setPipLayer] = useState<"raw" | "segmentation">("segmentation");
  const [msg, setMsg] = useState("");
  const [loading, setLoading] = useState(false);
  const [cachingMap, setCachingMap] = useState(false);
  const [showTiltDrawer, setShowTiltDrawer] = useState(false);
  const [showCoordsDrawer, setShowCoordsDrawer] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [cloudOpacity, setCloudOpacity] = useState<number>(0.88);
  const [savingGeom, setSavingGeom] = useState(false);
  const [geomDraft, setGeomDraft] = useState<CameraGeometryValues>({
    orientation_elevation_deg: 65,
    orientation_azimuth_deg: 0,
    orientation_roll_deg: 0,
    fov_horizontal_deg: 72,
    fov_vertical_deg: 45,
    assumed_cloud_altitude_m: null,
    tilt_correction_enabled: true,
  });

  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const leafletMapRef = useRef<L.Map | null>(null);
  const baseLayerGroupRef = useRef<L.LayerGroup | null>(null);
  const pinMarkerRef = useRef<L.Marker | null>(null);
  const rangeCircleRef = useRef<L.Circle | null>(null);
  const fovPolygonRef = useRef<L.Polygon | null>(null);
  const secondaryFovGroupRef = useRef<L.LayerGroup | null>(null);
  const cameraOverlayRef = useRef<L.ImageOverlay | null>(null);
  const regionalCloudOverlayRef = useRef<L.ImageOverlay | null>(null);
  const regionalCloudKeyRef = useRef<string>("");
  const rainLayerRef = useRef<L.TileLayer | null>(null);
  const trajectoryGroupRef = useRef<L.LayerGroup | null>(null);
  const hasCenteredRef = useRef<boolean>(false);
  const isLiveRef = useRef<boolean>(true);
  isLiveRef.current = isLive;
  const mapClickModeRef = useRef<"inspect" | "pin">("inspect");
  mapClickModeRef.current = mapClickMode;
  const stateRef = useRef<any>(null);
  stateRef.current = state;
  const selectedObsIdRef = useRef<number | null>(null);
  const rainTileUrlRef = useRef<string>("");
  const savePinRef = useRef<(lat: number, lon: number, desc?: string, radKm?: number) => Promise<void>>(
    async () => { }
  );

  const loadState = useCallback(async () => {
    setLoading(true);
    try {
      const s = await api.getRadarState();
      setState(s);
      if (s.location) {
        setLatInput(String(s.location.latitude));
        setLonInput(String(s.location.longitude));
        setRadiusInput(String(s.location.radar_radius_km ?? 15));
        setLocDesc(s.location.location_description || "");
        setGeomDraft({
          orientation_elevation_deg: Number(s.location.orientation_elevation_deg ?? 65),
          orientation_azimuth_deg: Number(s.location.orientation_azimuth_deg ?? 0),
          orientation_roll_deg: Number(s.location.orientation_roll_deg ?? 0),
          fov_horizontal_deg: Number(s.location.fov_horizontal_deg ?? 72),
          fov_vertical_deg: Number(s.location.fov_vertical_deg ?? 45),
          assumed_cloud_altitude_m: s.location.assumed_cloud_altitude_m ?? null,
          tilt_correction_enabled: Boolean(s.location.tilt_correction_enabled ?? true),
        });
      }
      const camHist = Array.isArray(s.camera_history) ? s.camera_history : [];
      const rvFrames = s.rainviewer?.radar_frames || [];
      const len = camHist.length > 0 ? camHist.length : rvFrames.length;
      if (len > 0) {
        if (isLiveRef.current) {
          setTimelineIdx(len - 1);
        } else if (camHist.length > 0 && selectedObsIdRef.current != null) {
          const foundIdx = camHist.findIndex((o: any) => o.id === selectedObsIdRef.current);
          setTimelineIdx(foundIdx >= 0 ? foundIdx : len - 1);
        } else {
          setTimelineIdx((prev) => Math.min(prev, len - 1));
        }
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadState();
    const id = window.setInterval(loadState, 15000);
    return () => window.clearInterval(id);
  }, [loadState, offlineMode]);

  const centerMapOn = useCallback((lat: number, lon: number, radKm: number, preserveHigherZoom = true) => {
    const map = leafletMapRef.current;
    if (!map) return;
    const targetZoom = zoomForRadius(radKm);
    const nextZoom = preserveHigherZoom ? Math.max(map.getZoom(), targetZoom) : targetZoom;
    map.setView([lat, lon], nextZoom, { animate: true });
  }, []);

  const savePinCoordinates = useCallback(
    async (lat: number, lon: number, desc?: string, radKm?: number) => {
      const targetRad = radKm ?? (Number(radiusInput) || 15);
      const res = await api.updateRadarLocation({
        latitude: Number(lat.toFixed(5)),
        longitude: Number(lon.toFixed(5)),
        location_description: desc !== undefined ? desc : "",
        radar_radius_km: targetRad,
      });
      setLatInput(String(res.latitude));
      setLonInput(String(res.longitude));
      setLocDesc(res.location_description || "");
      centerMapOn(res.latitude, res.longitude, res.radar_radius_km ?? targetRad, true);
      const baseMsg = t("radar.pinSaved", { lat: res.latitude.toFixed(4), lon: res.longitude.toFixed(4) });
      setMsg(res.location_description ? `${res.location_description} · ${baseMsg}` : baseMsg);
      setTimeout(() => setMsg(""), 3000);
      await loadState();
      onLocationSaved();
    },
    [centerMapOn, loadState, onLocationSaved, radiusInput, t]
  );

  savePinRef.current = savePinCoordinates;

  // Initialize Leaflet ONCE we have real saved coordinates — persists across Offline <-> Online switches
  useEffect(() => {
    if (!mapContainerRef.current || leafletMapRef.current || !state?.location) return;
    const lat = Number(state.location.latitude);
    const lon = Number(state.location.longitude);
    const radKm = Number(state.location.radar_radius_km || 15);

    const map = L.map(mapContainerRef.current, {
      center: [lat, lon],
      zoom: zoomForRadius(radKm),
      zoomControl: true,
    });
    hasCenteredRef.current = true;

    baseLayerGroupRef.current = L.layerGroup().addTo(map);
    trajectoryGroupRef.current = L.layerGroup().addTo(map);
    secondaryFovGroupRef.current = L.layerGroup().addTo(map);
    map.on("click", (evt: L.LeafletMouseEvent) => {
      if (mapClickModeRef.current === "pin") {
        savePinRef.current(evt.latlng.lat, evt.latlng.lng);
        return;
      }
      const st = stateRef.current;
      const camLat = Number(st?.location?.latitude ?? lat);
      const camLon = Number(st?.location?.longitude ?? lon);
      const dLatKm = (evt.latlng.lat - camLat) * 111.32;
      const dLonKm = (evt.latlng.lng - camLon) * 111.32 * Math.max(0.15, Math.cos((camLat * Math.PI) / 180));
      const distKm = Math.hypot(dLatKm, dLonKm);
      const bearingDeg = ((Math.atan2(dLonKm, dLatKm) * 180) / Math.PI + 360) % 360;
      const cardinal = degToCardinal(bearingDeg);
      const inFov = pointInPolygon(evt.latlng.lat, evt.latlng.lng, st?.camera_projection?.fov_polygon || []);

      // Sample nearest regional Open-Meteo grid point if available
      let nwpCloudPct: number | null = null;
      const rGrid = st?.regional_cloud_grid || [];
      if (Array.isArray(rGrid) && rGrid.length > 0) {
        let best = rGrid[0];
        let bestD = Infinity;
        for (const p of rGrid) {
          const d = Math.hypot(Number(p.latitude) - evt.latlng.lat, Number(p.longitude) - evt.latlng.lng);
          if (d < bestD) {
            bestD = d;
            best = p;
          }
        }
        if (best && best.cloud_cover_pct != null) {
          nwpCloudPct = Number(best.cloud_cover_pct);
        }
      }

      setInspectedPoint({
        lat: evt.latlng.lat,
        lon: evt.latlng.lng,
        distKm: Number(distKm.toFixed(2)),
        bearingDeg: Math.round(bearingDeg),
        cardinal,
        inFov,
        nwpCloudPct,
      });
    });

    leafletMapRef.current = map;
  }, [state?.location]);

  // Cleanup Leaflet only on component unmount
  useEffect(() => {
    return () => {
      if (leafletMapRef.current) {
        leafletMapRef.current.remove();
        leafletMapRef.current = null;
      }
      baseLayerGroupRef.current = null;
      trajectoryGroupRef.current = null;
      secondaryFovGroupRef.current = null;
      pinMarkerRef.current = null;
      rangeCircleRef.current = null;
      fovPolygonRef.current = null;
      cameraOverlayRef.current = null;
      regionalCloudOverlayRef.current = null;
      regionalCloudKeyRef.current = "";
      rainLayerRef.current = null;
      rainTileUrlRef.current = "";
      hasCenteredRef.current = false;
    };
  }, []);

  const cameraHistory: any[] = Array.isArray(state?.camera_history) ? state.camera_history : [];
  const rainFrames: any[] = state?.rainviewer?.radar_frames || [];
  const hasLocalHistory = cameraHistory.length > 0;
  const timelineLen = hasLocalHistory ? cameraHistory.length : rainFrames.length;
  const safeTimelineIdx = timelineLen > 0 ? Math.min(Math.max(0, timelineIdx), timelineLen - 1) : 0;
  const activeObs = hasLocalHistory ? cameraHistory[safeTimelineIdx] : state?.observation ?? null;
  selectedObsIdRef.current = activeObs?.id ?? null;
  const activeTimeSec: number | null = hasLocalHistory
    ? (activeObs?.time ?? null)
    : (rainFrames[safeTimelineIdx]?.time ?? null);
  const matchedRainFrame = findNearestRainFrame(rainFrames, activeTimeSec);

  useEffect(() => {
    if (!isPlaying || timelineLen <= 1) return;
    const id = window.setInterval(() => {
      setIsLive(false);
      setTimelineIdx((prev) => (prev + 1 >= timelineLen ? 0 : prev + 1));
    }, 450);
    return () => window.clearInterval(id);
  }, [isPlaying, timelineLen]);

  // Load locally cached/proxied map tiles (serves disk cache offline with zero external requests)
  const hasLocation = Boolean(state?.location);
  useEffect(() => {
    const grp = baseLayerGroupRef.current;
    if (!grp || !hasLocation) return;
    grp.clearLayers();
    if (mapStyle === "osm") {
      L.tileLayer("/api/radar/tiles/osm/{z}/{x}/{y}.png", {
        maxZoom: 18,
        attribution: "&copy; OpenStreetMap",
      }).addTo(grp);
    } else if (mapStyle === "satellite") {
      L.tileLayer("/api/radar/tiles/sat_base/{z}/{x}/{y}.png", {
        maxZoom: 18,
        attribution: "Tiles &copy; Esri",
      }).addTo(grp);
      L.tileLayer("/api/radar/tiles/dark_ref/{z}/{x}/{y}.png", {
        maxZoom: 18,
      }).addTo(grp);
    } else {
      L.tileLayer("/api/radar/tiles/dark_base/{z}/{x}/{y}.png", {
        maxZoom: 18,
        attribution: "Tiles &copy; Esri",
      }).addTo(grp);
      L.tileLayer("/api/radar/tiles/dark_ref/{z}/{x}/{y}.png", {
        maxZoom: 18,
      }).addTo(grp);
    }
  }, [mapStyle, hasLocation, offlineMode]);

  // Update pin, camera dome, motion vector, and optional online layers
  useEffect(() => {
    const map = leafletMapRef.current;
    if (!map || !state?.location) return;

    const lat = Number(state.location.latitude);
    const lon = Number(state.location.longitude);
    const radKm = Number(state.location.radar_radius_km || 15);

    if (!hasCenteredRef.current) {
      hasCenteredRef.current = true;
      map.setView([lat, lon], zoomForRadius(radKm));
    }

    if (!pinMarkerRef.current) {
      const pulseIcon = L.divIcon({
        className: "",
        html: `<div class="radar-pin-wrap"><div class="radar-pin-pulse"></div><div class="radar-pin-core"></div></div>`,
        iconSize: [22, 22],
        iconAnchor: [11, 11],
      });
      pinMarkerRef.current = L.marker([lat, lon], { icon: pulseIcon, zIndexOffset: 600 }).addTo(map);
    } else {
      pinMarkerRef.current.setLatLng([lat, lon]);
    }
    pinMarkerRef.current.bindTooltip(
      `${state.location.location_description || t("radar.yourPin")} (${lat.toFixed(4)}, ${lon.toFixed(4)})`,
      { direction: "top", offset: [0, -10] }
    );

    if (!rangeCircleRef.current) {
      rangeCircleRef.current = L.circle([lat, lon], {
        radius: radKm * 1000,
        color: "#38BDF8",
        weight: 1.5,
        dashArray: "5 5",
        fillOpacity: 0.03,
      }).addTo(map);
    } else {
      rangeCircleRef.current.setLatLng([lat, lon]);
      rangeCircleRef.current.setRadius(radKm * 1000);
    }

    if (!offlineMode && showRegionalClouds && Array.isArray(state.regional_cloud_bounds)) {
      const rKey = `${lat.toFixed(3)},${lon.toFixed(3)},${state?.online_weather?.fetched_at || ""}`;
      const rUrl = `/api/radar/regional-clouds.png?lat=${lat.toFixed(3)}&lon=${lon.toFixed(3)}&k=${encodeURIComponent(rKey)}`;
      const rBounds = L.latLngBounds(state.regional_cloud_bounds as L.LatLngTuple[]);
      if (!regionalCloudOverlayRef.current) {
        regionalCloudOverlayRef.current = L.imageOverlay(rUrl, rBounds, {
          opacity: 0.82,
          interactive: false,
        }).addTo(map);
        regionalCloudKeyRef.current = rKey;
      } else if (regionalCloudKeyRef.current !== rKey) {
        regionalCloudOverlayRef.current.setUrl(rUrl);
        regionalCloudOverlayRef.current.setBounds(rBounds);
        regionalCloudKeyRef.current = rKey;
      }
    } else if (regionalCloudOverlayRef.current) {
      map.removeLayer(regionalCloudOverlayRef.current);
      regionalCloudOverlayRef.current = null;
      regionalCloudKeyRef.current = "";
    }

    if (fovPolygonRef.current) {
      map.removeLayer(fovPolygonRef.current);
      fovPolygonRef.current = null;
    }
    if (showCameraDome && Array.isArray(state.camera_projection?.fov_polygon) && state.camera_projection.fov_polygon.length >= 3) {
      fovPolygonRef.current = L.polygon(state.camera_projection.fov_polygon, {
        color: "#38BDF8",
        weight: 1.75,
        dashArray: "4 4",
        fillColor: "#38BDF8",
        fillOpacity: 0.06,
        interactive: false,
      }).addTo(map);
    }
    if (secondaryFovGroupRef.current) {
      secondaryFovGroupRef.current.clearLayers();
      if (showCameraDome && Array.isArray(state.secondary_camera_projections)) {
        for (const sec of state.secondary_camera_projections) {
          if (Array.isArray(sec.fov_polygon) && sec.fov_polygon.length >= 3) {
            L.polygon(sec.fov_polygon, {
              color: "#34D399",
              weight: 1.75,
              dashArray: "5 4",
              fillColor: "#34D399",
              fillOpacity: 0.06,
            })
              .bindTooltip(
                `${sec.name} · Az ${Math.round(Number(sec.azimuth_deg ?? 0))}°${
                  sec.cloud_coverage_pct != null ? ` · ${sec.cloud_coverage_pct}%` : ""
                }`
              )
              .addTo(secondaryFovGroupRef.current);
          }
        }
      }
    }
    if (showCameraDome && activeObs) {
      const rawBounds = (state.camera_projection?.overlay_bounds || kmToBounds(lat, lon, radKm)) as L.LatLngTuple[];
      const bounds = L.latLngBounds(rawBounds);
      const obsQuery = activeObs.id ? `&obs_id=${activeObs.id}` : `&t=${encodeURIComponent(activeObs.captured_at)}`;
      const url = `/api/radar/cloud-overlay.png?size=512&el=${state.location.orientation_elevation_deg ?? 65}&az=${state.location.orientation_azimuth_deg ?? 0}&roll=${state.location.orientation_roll_deg ?? 0}&tc=${state.location.tilt_correction_enabled ? 1 : 0}${obsQuery}`;
      if (cameraOverlayRef.current) {
        cameraOverlayRef.current.setUrl(url);
        cameraOverlayRef.current.setBounds(bounds);
        cameraOverlayRef.current.setOpacity(cloudOpacity);
      } else {
        cameraOverlayRef.current = L.imageOverlay(url, bounds, { opacity: cloudOpacity, interactive: false }).addTo(map);
      }
    } else if (cameraOverlayRef.current) {
      map.removeLayer(cameraOverlayRef.current);
      cameraOverlayRef.current = null;
    }

    if (trajectoryGroupRef.current) {
      trajectoryGroupRef.current.clearLayers();
    }
    if (activeObs && activeObs.motion_speed_px_s > 0.2 && trajectoryGroupRef.current) {
      const tGrp = trajectoryGroupRef.current;
      const dirDeg = Number(activeObs.motion_direction_deg || 0);
      const deg = (dirDeg * Math.PI) / 180;
      const anchor = Array.isArray(state.camera_projection?.view_center_latlon)
        ? state.camera_projection.view_center_latlon
        : [lat, lon];
      const aLat = Number(anchor[0]);
      const aLon = Number(anchor[1]);
      const cosLat = Math.max(0.15, Math.cos((aLat * Math.PI) / 180));

      const cloudAltM = Number(state.camera_projection?.cloud_altitude_m || 2000);
      const degPerSec = Number(activeObs.motion_speed_deg_s ?? (activeObs.motion_speed_px_s * 72) / 1920);
      const speedKmh = Math.max(2.0, Math.min(130.0, ((degPerSec * Math.PI) / 180) * cloudAltM * 3.6));

      const projectOffset = (bearingRad: number, km: number): [number, number] => [
        aLat + (Math.cos(bearingRad) * km) / 111.32,
        aLon + (Math.sin(bearingRad) * km) / (111.32 * cosLat),
      ];

      // Scale visual trajectory so +5m, +15m, +20m (+30m online) waypoints are always legible inside the radar view
      const maxTrajMin = offlineMode ? 20 : 30;
      const rawMaxKm = (speedKmh * maxTrajMin) / 60;
      const visualMaxKm = Math.max(radKm * 0.45, Math.min(radKm * 1.45, rawMaxKm));
      const spreadRad = (12 * Math.PI) / 180;

      // +/- 12 deg arrival cone
      const coneLeft = projectOffset(deg - spreadRad, visualMaxKm);
      const coneRight = projectOffset(deg + spreadRad, visualMaxKm);
      L.polygon([[aLat, aLon], coneLeft, coneRight], {
        color: "#22D3EE",
        weight: 1,
        dashArray: "3 3",
        fillColor: "#22D3EE",
        fillOpacity: 0.08,
        interactive: false,
      }).addTo(tGrp);

      // Center trajectory line
      const endMax = projectOffset(deg, visualMaxKm);
      L.polyline([[aLat, aLon], endMax], {
        color: "#22D3EE",
        weight: 2.8,
        opacity: 0.92,
      }).addTo(tGrp);

      // +5m, +15m, +20m (and +30m when external NWP is active) projected markers
      const trajSteps = offlineMode ? [5, 15, 20] : [5, 15, 20, 30];
      for (const mins of trajSteps) {
        const frac = mins / maxTrajMin;
        const pt = projectOffset(deg, visualMaxKm * frac);
        const trueKm = (speedKmh * mins) / 60;
        L.circleMarker(pt, {
          radius: mins === maxTrajMin ? 5 : 4,
          color: "#08090B",
          weight: 1.5,
          fillColor: "#22D3EE",
          fillOpacity: 1,
        })
          .bindTooltip(`+${mins}m (~${trueKm.toFixed(1)} km · ${speedKmh.toFixed(1)} km/h)`, {
            direction: "top",
          })
          .addTo(tGrp);
      }
    }

    const targetRainUrl = !offlineMode && showRainRadar ? matchedRainFrame?.tile_url || "" : "";
    if (!targetRainUrl) {
      if (rainLayerRef.current) {
        map.removeLayer(rainLayerRef.current);
        rainLayerRef.current = null;
      }
      rainTileUrlRef.current = "";
    } else if (rainTileUrlRef.current !== targetRainUrl) {
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
      rainTileUrlRef.current = targetRainUrl;
    }
  }, [offlineMode, state, activeObs, matchedRainFrame?.tile_url, showCameraDome, showRegionalClouds, showRainRadar, cloudOpacity, t]);

  useEffect(() => {
    const id = window.setTimeout(() => {
      leafletMapRef.current?.invalidateSize({ animate: false });
    }, 80);
    return () => window.clearTimeout(id);
  }, [isFullscreen]);

  const handleUseGps = () => {
    if (!navigator.geolocation) {
      setMsg(t("radar.gpsUnsupported"));
      return;
    }
    if (!window.isSecureContext && !["localhost", "127.0.0.1"].includes(window.location.hostname)) {
      setMsg("Browser GPS requires HTTPS or localhost. Use 'Move Pin' on the map or City Search instead.");
      setMapClickMode("pin");
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const lat = pos.coords.latitude;
        const lon = pos.coords.longitude;
        savePinCoordinates(lat, lon);
      },
      (err) => {
        setMsg(err.message || t("radar.gpsUnsupported"));
        setMapClickMode("pin");
      },
      { enableHighAccuracy: true, timeout: 8000 }
    );
  };

  const handleCitySearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (offlineMode || searchQ.trim().length < 2) return;
    const res = await api.searchLocation(searchQ.trim(), locale);
    setSearchResults(res.results || []);
  };

  const handleCacheOfflineMap = async () => {
    if (offlineMode) return;
    setCachingMap(true);
    try {
      const res = await api.prefetchOfflineMapTiles({
        latitude: Number(state?.location?.latitude ?? latInput) || undefined,
        longitude: Number(state?.location?.longitude ?? lonInput) || undefined,
      });
      setMsg(
        t("radar.cachedSummary", {
          n: res.cached ?? 0,
          total: res.tile_cache?.cached_tiles ?? 0,
        })
      );
      setTimeout(() => setMsg(""), 4000);
      loadState();
    } catch (err: any) {
      setMsg(err.message);
    } finally {
      setCachingMap(false);
    }
  };

  const handleApplyGeometry = async () => {
    setSavingGeom(true);
    try {
      await api.updateCameraGeometry({
        ...geomDraft,
        recalculate_latest: true,
      });
      setMsg(t("tilt.appliedMsg"));
      setTimeout(() => setMsg(""), 3500);
      await loadState();
      onLocationSaved();
    } catch (err: any) {
      setMsg(err.message);
    } finally {
      setSavingGeom(false);
    }
  };

  const obs = activeObs;
  const wx = state?.online_weather?.current;
  const cachedTiles = Number(state?.tile_cache?.cached_tiles ?? 0);

  return (
    <div className="space-y-4">
      {/* Header + Mode Banner */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-white flex items-center gap-2">
            <MapPin className="w-4 h-4 text-sky-400" /> {t("radar.title")}
          </h2>
          <p className="text-xs text-[#8E95A2]">{t("radar.subtitle")}</p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => onToggleMode(true)}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition ${offlineMode
              ? "bg-emerald-500/15 border-emerald-500/40 text-emerald-300"
              : "obs-btn-secondary"
              }`}
          >
            <Lock className="w-3.5 h-3.5" /> {t("mode.offlineOnly")}
          </button>
          <button
            type="button"
            onClick={() => onToggleMode(false)}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition ${!offlineMode
              ? "bg-sky-500/20 border-sky-400/40 text-sky-200"
              : "obs-btn-secondary"
              }`}
          >
            <Globe className="w-3.5 h-3.5" /> {t("mode.onlineHybrid")}
          </button>
          {!offlineMode && (
            <button
              type="button"
              disabled={cachingMap}
              onClick={handleCacheOfflineMap}
              className="obs-btn-secondary inline-flex items-center gap-1.5 px-3 py-1.5 text-xs"
            >
              <Download className="w-3.5 h-3.5 text-emerald-400" />
              {cachingMap ? t("radar.cachingOffline") : t("radar.cacheOfflineBtn")}
            </button>
          )}
          <button
            type="button"
            onClick={() => setShowTiltDrawer((v) => !v)}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition ${showTiltDrawer
              ? "bg-sky-500/20 border-sky-400/50 text-sky-200"
              : "obs-btn-secondary"
              }`}
          >
            <Sliders className="w-3.5 h-3.5 text-sky-400" />
            {t("tilt.radarBtn")} ({Math.round(90 - geomDraft.orientation_elevation_deg)}° · {Math.round(geomDraft.orientation_azimuth_deg)}°)
          </button>
          <button
            type="button"
            onClick={loadState}
            className="obs-btn-secondary inline-flex items-center gap-1 px-2.5 py-1.5 text-xs"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
      </div>

      {msg && <div className="obs-inset px-4 py-2 text-xs text-sky-200">{msg}</div>}

      {showTiltDrawer && (
        <CameraTiltCalibrator
          value={geomDraft}
          onChange={setGeomDraft}
          onApply={handleApplyGeometry}
          applying={savingGeom}
          compact
          solar={state?.camera_projection?.solar || null}
          effectiveCloudAltitudeM={Number(state?.camera_projection?.cloud_altitude_m || 2000)}
        />
      )}

      {/* Compact Geolocation & Search Strip (with collapsible manual coordinates) */}
      <div className="obs-panel p-3 space-y-2.5 text-xs">
        <div className="flex flex-wrap items-center justify-between gap-2">
          {/* City Search or Offline Status */}
          <div className="flex items-center gap-2 flex-1 min-w-[220px]">
            {!offlineMode ? (
              <form onSubmit={handleCitySearch} className="flex gap-1.5 flex-1 max-w-md">
                <div className="relative flex-1">
                  <Search className="w-3.5 h-3.5 text-[#64748B] absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
                  <input
                    type="text"
                    value={searchQ}
                    onChange={(e) => setSearchQ(e.target.value)}
                    placeholder={t("radar.searchPlaceholder")}
                    className="obs-input w-full pl-8 py-1.5"
                  />
                </div>
                <button
                  type="submit"
                  className="obs-btn-secondary inline-flex items-center gap-1 px-3 py-1.5"
                >
                  {t("radar.searchBtn")}
                </button>
              </form>
            ) : (
              <div className="text-[11px] text-emerald-400 flex items-center gap-1.5 py-1">
                <Lock className="w-3.5 h-3.5 shrink-0" />
                <span>
                  {cachedTiles > 0
                    ? t("radar.offlineCachedPill", { n: cachedTiles })
                    : t("radar.offlineNeedCache")}
                </span>
              </div>
            )}
          </div>

          {/* Quick Actions: Center, GPS, Manual Coordinates Toggle */}
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              type="button"
              onClick={() => {
                if (state?.location) {
                  centerMapOn(
                    Number(state.location.latitude),
                    Number(state.location.longitude),
                    Number(state.location.radar_radius_km || 15),
                    false
                  );
                }
              }}
              title={t("radar.centerPin")}
              className="obs-btn-secondary inline-flex items-center gap-1.5 px-2.5 py-1.5"
            >
              <LocateFixed className="w-3.5 h-3.5 text-sky-400" />
              <span className="hidden sm:inline">{t("radar.centerPin")}</span>
            </button>
            <button
              type="button"
              onClick={handleUseGps}
              className="obs-btn-secondary inline-flex items-center gap-1.5 px-2.5 py-1.5"
            >
              <Crosshair className="w-3.5 h-3.5 text-emerald-400" />
              <span className="hidden sm:inline">{t("radar.useGps")}</span>
            </button>
            <button
              type="button"
              onClick={() => setShowCoordsDrawer((v) => !v)}
              className={`obs-btn-secondary inline-flex items-center gap-1 px-2.5 py-1.5 font-mono text-[11px] ${
                showCoordsDrawer ? "border-sky-400/40 text-sky-200 bg-sky-500/15" : ""
              }`}
            >
              <span>
                {latInput && lonInput ? `${Number(latInput).toFixed(2)}°, ${Number(lonInput).toFixed(2)}° · ${radiusInput}km` : t("radar.setPin")}
              </span>
              {showCoordsDrawer ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>

        {/* Collapsible Manual Coordinates Form */}
        {showCoordsDrawer && (
          <div className="pt-2.5 border-t border-white/[0.07] flex flex-wrap items-end gap-2">
            <label className="space-y-1">
              <span className="obs-label block">{t("radar.lat")}</span>
              <input
                type="number"
                step="0.0001"
                value={latInput}
                onChange={(e) => setLatInput(e.target.value)}
                className="obs-input w-28 font-mono tabular-nums"
              />
            </label>
            <label className="space-y-1">
              <span className="obs-label block">{t("radar.lon")}</span>
              <input
                type="number"
                step="0.0001"
                value={lonInput}
                onChange={(e) => setLonInput(e.target.value)}
                className="obs-input w-28 font-mono tabular-nums"
              />
            </label>
            <label className="space-y-1">
              <span className="obs-label block">{t("radar.radiusKm")}</span>
              <input
                type="number"
                min={1}
                max={100}
                value={radiusInput}
                onChange={(e) => setRadiusInput(e.target.value)}
                className="obs-input w-20 font-mono tabular-nums"
              />
            </label>
            <button
              type="button"
              onClick={() => {
                const la = Number(latInput);
                const lo = Number(lonInput);
                const rad = Number(radiusInput) || 15;
                savePinCoordinates(la, lo, locDesc, rad);
                setShowCoordsDrawer(false);
              }}
              className="obs-btn-primary px-3.5 py-1.5"
            >
              {t("radar.setPin")}
            </button>
          </div>
        )}

        {searchResults.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pt-2 border-t border-white/[0.06]">
            {searchResults.map((r, i) => (
              <button
                key={i}
                type="button"
                onClick={() => {
                  const label = `${r.name}${r.admin1 ? `, ${r.admin1}` : ""}`;
                  setLocDesc(label);
                  setSearchResults([]);
                  savePinCoordinates(r.latitude, r.longitude, label);
                }}
                className="obs-btn-secondary px-2.5 py-1 text-sky-300"
              >
                {r.name} {r.admin1 ? `(${r.admin1}, ${r.country})` : `(${r.country})`}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Main Interactive Geo Radar Map + Side Telemetry */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-4">
        <div
          className={
            isFullscreen
              ? "fixed inset-3 z-50 obs-panel overflow-hidden flex flex-col shadow-[0_24px_80px_rgba(0,0,0,0.85)]"
              : "xl:col-span-3 obs-panel overflow-hidden flex flex-col"
          }
        >
          <div className={`relative w-full bg-[#060911] ${isFullscreen ? "flex-1 min-h-0" : "h-[580px] lg:h-[660px]"}`}>
            <div ref={mapContainerRef} className="w-full h-full z-0" />

            {/* Top-Left Windy-Style Meteorological Cloud Scale & Legend */}
            <div className="absolute top-3 left-14 z-[500] obs-glass-pill px-3 py-1.5 flex flex-wrap items-center gap-3 text-[10px] font-mono text-[#D2D6DF] pointer-events-none">
              <div className="flex items-center gap-2">
                <div className="w-20 h-2 rounded-full bg-gradient-to-r from-[#38BDF8] via-[#F8FAFC] to-[#C34BEB] border border-white/15" />
                <span className="text-sky-300">{t("radar.legendThin")}</span>
                <span className="text-white">{t("radar.legendDense")}</span>
                <span className="text-fuchsia-300">{t("radar.legendDark")}</span>
              </div>
              <span className="inline-flex items-center gap-1.5 text-amber-300">
                <span className="w-2 h-2 rounded-full bg-[#FBBF24] inline-block" /> {t("vision.legend.sunGlare")}
              </span>
              <span className="inline-flex items-center gap-1.5 text-sky-300">
                <span className="w-2.5 h-2.5 rounded-sm border border-dashed border-sky-400 bg-sky-400/20 inline-block" /> {t("radar.legendFov")}
              </span>
              <span className="inline-flex items-center gap-1.5 text-cyan-300">
                <span className="w-3 h-0.5 bg-cyan-400 inline-block rounded-full" /> {t("radar.legendTraj")}
              </span>
            </div>

            {/* Top-Right Click Mode Switcher, Fullscreen Button & Point Inspection Card */}
            <div className="absolute top-3 right-3 z-[500] flex flex-col items-end gap-2">
              <div className="flex items-center gap-1.5">
                <div className="obs-glass-pill p-1 inline-flex items-center gap-1 text-[11px] font-mono">
                  <button
                    type="button"
                    onClick={() => setMapClickMode("inspect")}
                    className={`px-2.5 py-1 rounded-lg transition ${mapClickMode === "inspect"
                      ? "bg-sky-500/25 border border-sky-400/50 text-sky-200 font-semibold"
                      : "text-[#9AA1AD] hover:text-white border border-transparent"
                      }`}
                  >
                    {t("radar.modeInspect")}
                  </button>
                  <button
                    type="button"
                    onClick={() => setMapClickMode("pin")}
                    className={`px-2.5 py-1 rounded-lg transition ${mapClickMode === "pin"
                      ? "bg-emerald-500/25 border border-emerald-400/50 text-emerald-200 font-semibold"
                      : "text-[#9AA1AD] hover:text-white border border-transparent"
                      }`}
                  >
                    {t("radar.modeMovePin")}
                  </button>
                </div>
                <button
                  type="button"
                  onClick={() => setIsFullscreen((v) => !v)}
                  title={isFullscreen ? "Exit Fullscreen" : "Fullscreen Radar"}
                  className="obs-glass-pill p-2 text-slate-200 hover:text-sky-300 hover:border-sky-400/40 transition"
                >
                  {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
                </button>
              </div>

              {inspectedPoint && (
                <div className="w-64 bg-[#0B0C0E]/95 backdrop-blur border border-sky-400/30 rounded-xl p-3 text-xs space-y-1.5 shadow-xl">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-white">{t("radar.inspectTitle")}</span>
                    <button
                      type="button"
                      onClick={() => setInspectedPoint(null)}
                      className="text-[#8E95A2] hover:text-white text-[11px] font-mono"
                    >
                      ✕
                    </button>
                  </div>
                  <div className="font-mono text-[11px] text-[#9AA1AD]">
                    {inspectedPoint.lat.toFixed(4)}°, {inspectedPoint.lon.toFixed(4)}°
                  </div>
                  <div className="font-mono text-[11px] text-sky-300">
                    {t("radar.inspectDist", {
                      km: inspectedPoint.distKm,
                      deg: inspectedPoint.bearingDeg,
                      dir: tCardinal(inspectedPoint.cardinal),
                    })}
                  </div>
                  <div className="text-[11px]">
                    <span
                      className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-mono border ${inspectedPoint.inFov
                        ? "bg-sky-500/15 border-sky-400/35 text-sky-300"
                        : "bg-white/[0.05] border-white/10 text-[#8E95A2]"
                        }`}
                    >
                      {inspectedPoint.inFov ? t("radar.inspectInFov") : t("radar.inspectOutFov")}
                    </span>
                  </div>
                  {inspectedPoint.nwpCloudPct != null && (
                    <div className="text-[11px] font-mono text-[#C5CBD5]">
                      Open-Meteo NWP: <strong>{inspectedPoint.nwpCloudPct}%</strong>
                    </div>
                  )}
                  <button
                    type="button"
                    onClick={() => {
                      const pt = inspectedPoint;
                      setInspectedPoint(null);
                      savePinCoordinates(pt.lat, pt.lon);
                    }}
                    className="obs-btn-primary w-full py-1 text-[11px] mt-1"
                  >
                    {t("radar.inspectMoveHere")}
                  </button>
                </div>
              )}
            </div>

            {/* Bottom-Left Picture-in-Picture (PiP) Sky Frame Card While Scrubbing Timeline */}
            {activeObs && (
              <div className="absolute bottom-3 left-3 z-[500]">
                {showPip ? (
                  <div className="w-64 sm:w-72 obs-glass-pill overflow-hidden text-[11px]">
                    <div className="px-2.5 py-1.5 border-b border-white/[0.08] flex items-center justify-between gap-1">
                      <span className="font-semibold text-white truncate">
                        {t("radar.pipTitle")} #{activeObs.id}
                      </span>
                      <div className="flex items-center gap-1">
                        <button
                          type="button"
                          onClick={() => setPipLayer(pipLayer === "raw" ? "segmentation" : "raw")}
                          className="px-1.5 py-0.5 rounded-md bg-white/[0.08] hover:bg-white/[0.15] text-[10px] font-mono text-sky-300 transition"
                        >
                          {pipLayer === "raw" ? t("radar.pipRaw") : t("radar.pipSeg")}
                        </button>
                        <button
                          type="button"
                          onClick={() => setShowPip(false)}
                          title={t("radar.pipHide")}
                          className="px-1 text-[#8E95A2] hover:text-white font-mono"
                        >
                          –
                        </button>
                      </div>
                    </div>
                    <div className="aspect-video bg-[#050608] relative">
                      <img
                        src={`/api/analysis/image/${activeObs.id}?layer=${pipLayer}`}
                        alt="PiP camera sky"
                        className="w-full h-full object-cover"
                      />
                      <div className="absolute bottom-1 left-1.5 right-1.5 px-1.5 py-0.5 rounded-md bg-black/75 backdrop-blur flex items-center justify-between text-[10px] font-mono text-white">
                        <span>
                          {activeTimeSec != null ? new Date(activeTimeSec * 1000).toLocaleTimeString() : ""}
                        </span>
                        <span className="text-sky-300 font-bold tabular-nums">
                          {Math.round((activeObs.cloud_coverage_pct / 100) * 8)}/8 · {activeObs.cloud_coverage_pct.toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={() => setShowPip(true)}
                    className="obs-glass-pill px-2.5 py-1.5 text-[11px] font-mono text-sky-300 hover:text-white transition"
                  >
                    {t("radar.pipShow")} #{activeObs.id}
                  </button>
                )}
              </div>
            )}

            {/* Bottom-Right Floating Windy-Style Basemap, Layers & Opacity Dock */}
            <div className="absolute bottom-7 right-3 z-[500] flex flex-col items-end gap-1.5 max-w-[calc(100%-24px)]">
              <div className="obs-glass-pill p-1.5 flex flex-wrap items-center gap-1.5 text-[11px]">
                {/* Basemap Style Segmented Pills */}
                <div className="inline-flex items-center bg-white/[0.04] rounded-lg p-0.5 border border-white/[0.06]">
                  {(
                    [
                      { id: "dark", label: t("radar.styleDark") },
                      { id: "osm", label: t("radar.styleStreet") },
                      { id: "satellite", label: t("radar.styleSat") },
                    ] as const
                  ).map((st) => (
                    <button
                      key={st.id}
                      type="button"
                      onClick={() => setMapStyle(st.id)}
                      className={`px-2 py-1 rounded-md text-[10.5px] font-medium transition ${
                        mapStyle === st.id
                          ? "bg-sky-400 text-[#050912] font-semibold shadow-sm"
                          : "text-[#94A3B8] hover:text-white"
                      }`}
                    >
                      {st.label}
                    </button>
                  ))}
                </div>

                {/* Layer Toggle Chips */}
                <button
                  type="button"
                  onClick={() => setShowCameraDome((v) => !v)}
                  className={`inline-flex items-center gap-1 px-2 py-1 rounded-lg border text-[10.5px] font-medium transition ${
                    showCameraDome
                      ? "bg-sky-500/20 border-sky-400/45 text-sky-200"
                      : "bg-white/[0.03] border-white/[0.08] text-[#8A96AA] hover:text-white"
                  }`}
                >
                  <Cloud className="w-3 h-3" />
                  <span>{t("radar.layerCameraClouds")}</span>
                </button>

                {!offlineMode && (
                  <>
                    <button
                      type="button"
                      onClick={() => setShowRegionalClouds((v) => !v)}
                      className={`inline-flex items-center gap-1 px-2 py-1 rounded-lg border text-[10.5px] font-medium transition ${
                        showRegionalClouds
                          ? "bg-indigo-500/20 border-indigo-400/45 text-indigo-200"
                          : "bg-white/[0.03] border-white/[0.08] text-[#8A96AA] hover:text-white"
                      }`}
                    >
                      <Globe className="w-3 h-3" />
                      <span>{t("radar.layerOpenMeteo")}</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowRainRadar((v) => !v)}
                      className={`inline-flex items-center gap-1 px-2 py-1 rounded-lg border text-[10.5px] font-medium transition ${
                        showRainRadar
                          ? "bg-cyan-500/20 border-cyan-400/45 text-cyan-200"
                          : "bg-white/[0.03] border-white/[0.08] text-[#8A96AA] hover:text-white"
                      }`}
                    >
                      <Droplets className="w-3 h-3" />
                      <span>{t("radar.layerRainViewer")}</span>
                    </button>
                  </>
                )}

                {/* Cloud Overlay Opacity Slider */}
                {showCameraDome && (
                  <div className="inline-flex items-center gap-1.5 pl-1.5 pr-1 border-l border-white/[0.1] text-[10px] font-mono text-[#94A3B8]">
                    <Eye className="w-3 h-3 text-sky-300 shrink-0" />
                    <input
                      type="range"
                      min={25}
                      max={100}
                      value={Math.round(cloudOpacity * 100)}
                      onChange={(e) => setCloudOpacity(Number(e.target.value) / 100)}
                      title="Cloud Overlay Opacity"
                      className="w-16 h-1 cursor-pointer"
                    />
                    <span className="w-7 text-right tabular-nums text-sky-200">
                      {Math.round(cloudOpacity * 100)}%
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {(hasLocalHistory || (!offlineMode && showRainRadar && rainFrames.length > 0)) && timelineLen > 0 && (
            <div className="px-4 py-2.5 border-t border-white/[0.07] bg-[#101217] space-y-2 text-xs">
              {/* Color-Coded Historical Cloud Cover Sparkline Track */}
              {hasLocalHistory && cameraHistory.length > 1 && (
                <div className="flex items-end gap-[2px] h-5 w-full px-0.5 select-none">
                  {cameraHistory.map((hItem, idx) => {
                    const cov = Number(hItem.cloud_coverage_pct ?? 0);
                    const dark = Number(hItem.dark_cloud_pct ?? 0);
                    const dense = Number(hItem.dense_cloud_pct ?? 0);
                    const isSelected = idx === safeTimelineIdx;
                    const barColor =
                      dark >= 20
                        ? "#F43F5E"
                        : dense >= 25
                          ? "#F8FAFC"
                          : cov >= 12
                            ? "#38BDF8"
                            : "#1E3A5F";
                    const heightPct = Math.max(22, Math.min(100, Math.round(20 + cov * 0.8)));
                    return (
                      <button
                        key={hItem.id || idx}
                        type="button"
                        onClick={() => {
                          setIsPlaying(false);
                          setTimelineIdx(idx);
                          setIsLive(idx >= timelineLen - 1);
                        }}
                        title={`${new Date((hItem.time || 0) * 1000).toLocaleTimeString()} · ${cov.toFixed(1)}%`}
                        style={{ height: `${heightPct}%`, backgroundColor: barColor }}
                        className={`flex-1 rounded-t-[2px] transition-opacity ${isSelected ? "ring-1 ring-sky-300 opacity-100" : "opacity-65 hover:opacity-100"
                          }`}
                      />
                    );
                  })}
                </div>
              )}

              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  disabled={timelineLen <= 1}
                  onClick={() => {
                    if (isPlaying) {
                      setIsPlaying(false);
                    } else {
                      if (safeTimelineIdx >= timelineLen - 1) {
                        setTimelineIdx(0);
                      }
                      setIsLive(false);
                      setIsPlaying(true);
                    }
                  }}
                  title={isPlaying ? t("radar.pause") : t("radar.play")}
                  className="obs-btn-secondary inline-flex items-center gap-1 px-2.5 py-1 text-[11px] disabled:opacity-40"
                >
                  {isPlaying ? <Pause className="w-3.5 h-3.5 text-sky-300" /> : <Play className="w-3.5 h-3.5 text-sky-300" />}
                  <span>{isPlaying ? t("radar.pause") : t("radar.play")}</span>
                </button>

                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold border ${isLive
                    ? "bg-emerald-500/15 border-emerald-400/35 text-emerald-300"
                    : "bg-amber-500/15 border-amber-400/35 text-amber-300"
                    }`}
                >
                  {isLive ? t("radar.timelineLive") : t("radar.timelineHistory")}
                </span>

                <span className="text-[#C5CBD5] font-mono tabular-nums">
                  {activeTimeSec != null ? new Date(activeTimeSec * 1000).toLocaleTimeString() : "--:--:--"}
                </span>

                {hasLocalHistory && (
                  <span className="text-[11px] font-mono tabular-nums text-[#8E95A2]">
                    {t("radar.localFramesCount", { idx: safeTimelineIdx + 1, total: timelineLen })}
                  </span>
                )}

                <input
                  type="range"
                  min={0}
                  max={Math.max(0, timelineLen - 1)}
                  value={safeTimelineIdx}
                  onChange={(e) => {
                    const idx = Number(e.target.value);
                    setIsPlaying(false);
                    setTimelineIdx(idx);
                    setIsLive(idx >= timelineLen - 1);
                  }}
                  className="flex-1 min-w-[140px] accent-[#F4F5F7]"
                />

                {!isLive && (
                  <button
                    type="button"
                    onClick={() => {
                      setIsPlaying(false);
                      setIsLive(true);
                      setTimelineIdx(Math.max(0, timelineLen - 1));
                    }}
                    className="obs-btn-primary px-2.5 py-1 text-[11px]"
                  >
                    {t("radar.goLive")}
                  </button>
                )}

                <span className="text-[11px] text-[#646B78]">
                  {!offlineMode && showRainRadar && rainFrames.length > 0
                    ? "Weather data by RainViewer & Open-Meteo"
                    : t("radar.localOnlyAttribution")}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Side Observation & Online Weather Panel */}
        <RadarTelemetrySidebar
          obs={obs}
          state={state}
          wx={wx}
          isLive={isLive}
          activeTimeSec={activeTimeSec}
          offlineMode={offlineMode}
          onToggleMode={onToggleMode}
        />
      </div>
    </div>
  );
};
