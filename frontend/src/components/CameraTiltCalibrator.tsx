import React, { useMemo, useRef, useState } from "react";
import { Compass, Check, Eye, Sliders, Sun } from "lucide-react";
import { useI18n } from "../i18n/I18nContext";
import { SkyHorizonOverlay } from "./SkyHorizonOverlay";

export interface CameraGeometryValues {
  orientation_elevation_deg: number;
  orientation_azimuth_deg: number;
  orientation_roll_deg: number;
  fov_horizontal_deg: number;
  fov_vertical_deg: number;
  assumed_cloud_altitude_m: number | null;
  tilt_correction_enabled: boolean;
}

interface Props {
  value: CameraGeometryValues;
  onChange: (next: CameraGeometryValues) => void;
  onApply?: () => void | Promise<void>;
  applying?: boolean;
  compact?: boolean;
  solar?: {
    elevation_deg?: number;
    azimuth_deg?: number;
    is_daylight?: boolean;
    in_camera_fov?: boolean;
  } | null;
  effectiveCloudAltitudeM?: number;
}

const CARDINAL_ANGLES: { label: string; deg: number }[] = [
  { label: "N", deg: 0 },
  { label: "NE", deg: 45 },
  { label: "E", deg: 90 },
  { label: "SE", deg: 135 },
  { label: "S", deg: 180 },
  { label: "SO", deg: 225 },
  { label: "O", deg: 270 },
  { label: "NO", deg: 315 },
];

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

export const CameraTiltCalibrator: React.FC<Props> = ({
  value,
  onChange,
  onApply,
  applying = false,
  compact = false,
  solar = null,
  effectiveCloudAltitudeM,
}) => {
  const { t, tCardinal } = useI18n();
  const sideSvgRef = useRef<SVGSVGElement | null>(null);
  const compassSvgRef = useRef<SVGSVGElement | null>(null);
  const [draggingSide, setDraggingSide] = useState(false);
  const [draggingCompass, setDraggingCompass] = useState(false);
  const [showLiveCam, setShowLiveCam] = useState(!compact);

  const elDeg = clamp(Number(value.orientation_elevation_deg ?? 65), -20, 90);
  const tiltDeg = Math.round((90 - elDeg) * 10) / 10;
  const azDeg = ((Number(value.orientation_azimuth_deg ?? 0) % 360) + 360) % 360;
  const rollDeg = clamp(Number(value.orientation_roll_deg ?? 0), -45, 45);
  const fovH = clamp(Number(value.fov_horizontal_deg || 72), 10, 170);
  const fovV = clamp(Number(value.fov_vertical_deg || 45), 10, 170);
  const cloudAltM = clamp(Number(value.assumed_cloud_altitude_m || effectiveCloudAltitudeM || 2000), 100, 15000);

  const updateField = (patch: Partial<CameraGeometryValues>) => {
    onChange({
      ...value,
      ...patch,
    });
  };

  // Side-profile 2D SVG geometry (viewBox 560 x 270)
  const px = 128;
  const py = 218;
  const ceilingY = 62;
  const groundY = 228;

  const geom = useMemo(() => {
    const rad = (deg: number) => (deg * Math.PI) / 180;
    const elRad = rad(elDeg);
    const halfV = fovV / 2;
    const upperAngle = elDeg + halfV;
    const lowerAngle = elDeg - halfV;

    const traceRayToCeilingOrBox = (angleDeg: number, capR = 395): [number, number] => {
      const a = rad(angleDeg);
      const vx = Math.cos(a);
      const vy = -Math.sin(a); // SVG Y increases downward
      let tHit = capR;
      if (vy < -0.02) {
        const tCeil = (ceilingY - py) / vy;
        if (tCeil > 0) tHit = Math.min(tHit, tCeil);
      } else if (vy > 0.02) {
        const tGround = (groundY + 28 - py) / vy;
        if (tGround > 0) tHit = Math.min(tHit, tGround);
      }
      if (vx > 0.02) {
        tHit = Math.min(tHit, (544 - px) / vx);
      } else if (vx < -0.02) {
        tHit = Math.min(tHit, (20 - px) / vx);
      }
      return [px + vx * tHit, py + vy * tHit];
    };

    // Build Sky FOV Polygon (angles >= 0)
    const skyStart = Math.max(0, lowerAngle);
    const skyEnd = Math.max(skyStart, upperAngle);
    const skyPts: [number, number][] = [[px, py]];
    const samples = 16;
    for (let i = 0; i <= samples; i++) {
      const ang = skyStart + ((skyEnd - skyStart) * i) / samples;
      skyPts.push(traceRayToCeilingOrBox(ang, 410));
    }

    // Build Below-Horizon warning wedge if lower ray drops below 0 deg
    const belowHorizonPts: [number, number][] = [];
    if (lowerAngle < 0) {
      const clampedLow = Math.max(-38, lowerAngle);
      belowHorizonPts.push([px, py]);
      belowHorizonPts.push(traceRayToCeilingOrBox(0, 220));
      belowHorizonPts.push(traceRayToCeilingOrBox(clampedLow, 220));
    }

    const axisEnd = traceRayToCeilingOrBox(elDeg, 395);
    const handleDist = 112;
    const handleX = px + Math.cos(elRad) * handleDist;
    const handleY = py - Math.sin(elRad) * handleDist;

    // Angle arc between zenith (0, -r) and optical axis
    const arcR = 54;
    const zStartX = px;
    const zStartY = py - arcR;
    const axArcX = px + Math.cos(elRad) * arcR;
    const axArcY = py - Math.sin(elRad) * arcR;

    // Elevation arc from horizon (r, 0) to optical axis
    const elArcR = 72;
    const hStartX = px + elArcR;
    const hStartY = py;
    const elEndX = px + Math.cos(elRad) * elArcR;
    const elEndY = py - Math.sin(elRad) * elArcR;

    // Physical projected ground distances in km
    const minVisibleEl = Math.max(2, lowerAngle);
    const maxVisibleEl = Math.min(175, upperAngle);
    const altKm = cloudAltM / 1000;
    const centerDistKm = elDeg >= 88.5 ? 0 : Math.min(28, altKm / Math.tan(rad(Math.max(2, elDeg))));
    const nearDistKm =
      maxVisibleEl >= 90
        ? 0
        : altKm / Math.tan(rad(Math.max(4, maxVisibleEl)));
    const farDistKm =
      lowerAngle <= 4
        ? Math.min(28, altKm / Math.tan(rad(6)))
        : Math.min(28, altKm / Math.tan(rad(minVisibleEl)));
    const belowHorizonPct = lowerAngle < 0 ? Math.min(100, Math.round(((-lowerAngle) / fovV) * 100)) : 0;

    return {
      skyPolyStr: skyPts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" "),
      belowPolyStr: belowHorizonPts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" "),
      axisEnd,
      handleX,
      handleY,
      zStartX,
      zStartY,
      axArcX,
      axArcY,
      hStartX,
      hStartY,
      elEndX,
      elEndY,
      centerDistKm,
      nearDistKm: Math.min(nearDistKm, farDistKm),
      farDistKm: Math.max(nearDistKm, farDistKm),
      belowHorizonPct,
    };
  }, [cloudAltM, elDeg, fovV]);

  // Compass 2D top-down wedge geometry (viewBox 180 x 180)
  const compassWedge = useMemo(() => {
    const cx = 90;
    const cy = 90;
    const maxR = 66;
    const altKm = cloudAltM / 1000;
    const rad = (deg: number) => (deg * Math.PI) / 180;
    const centerDistKm = elDeg >= 89 ? 0 : Math.min(18, altKm / Math.tan(rad(Math.max(2, elDeg))));
    const spreadKm = Math.max(1.2, altKm * Math.tan(rad(Math.min(75, fovH / 2))));
    const mapScale = maxR / Math.max(6, centerDistKm + spreadKm);
    const rCenter = centerDistKm * mapScale;
    const rFront = Math.min(maxR, rCenter + spreadKm * 0.9 * mapScale);
    const rBack = Math.max(-22, rCenter - spreadKm * 0.65 * mapScale);
    const azRad = rad(azDeg);
    const fwX = Math.sin(azRad);
    const fwY = -Math.cos(azRad);
    const rtX = Math.cos(azRad);
    const rtY = Math.sin(azRad);
    const halfWFront = Math.min(maxR * 0.85, Math.max(12, spreadKm * mapScale * (1 + (90 - elDeg) * 0.012)));
    const halfWBack = Math.max(8, halfWFront * (elDeg >= 78 ? 0.9 : 0.45));

    const pt = (fwd: number, side: number): [number, number] => [
      cx + fwX * fwd + rtX * side,
      cy + fwY * fwd + rtY * side,
    ];
    const p1 = pt(rBack, -halfWBack);
    const p2 = pt(rFront, -halfWFront);
    const p3 = pt(rFront, halfWFront);
    const p4 = pt(rBack, halfWBack);
    const aim = pt(Math.min(maxR, rCenter), 0);

    return {
      poly: [p1, p2, p3, p4].map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" "),
      aim,
      dirEnd: pt(maxR + 4, 0),
    };
  }, [azDeg, cloudAltM, elDeg, fovH]);

  const handleSidePointer = (e: React.PointerEvent<SVGSVGElement>) => {
    const svg = sideSvgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 560;
    const y = ((e.clientY - rect.top) / rect.height) * 270;
    const dx = Math.max(0.01, x - px);
    const dy = py - y;
    const angle = Math.round((Math.atan2(dy, dx) * 180) / Math.PI);
    updateField({ orientation_elevation_deg: clamp(angle, -20, 90) });
  };

  const handleCompassPointer = (e: React.PointerEvent<SVGSVGElement>) => {
    const svg = compassSvgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 180;
    const y = ((e.clientY - rect.top) / rect.height) * 180;
    const deg = Math.round(((Math.atan2(x - 90, 90 - y) * 180) / Math.PI + 360) % 360);
    updateField({ orientation_azimuth_deg: deg });
  };

  const cardLabel =
    CARDINAL_ANGLES.find((c) => Math.abs(((azDeg - c.deg + 540) % 360) - 180) <= 22.5)?.label === "SO"
      ? tCardinal("SW")
      : CARDINAL_ANGLES.find((c) => Math.abs(((azDeg - c.deg + 540) % 360) - 180) <= 22.5)?.label === "O"
      ? tCardinal("W")
      : CARDINAL_ANGLES.find((c) => Math.abs(((azDeg - c.deg + 540) % 360) - 180) <= 22.5)?.label === "NO"
      ? tCardinal("NW")
      : tCardinal(
          (["N", "NE", "E", "SE", "S", "SW", "W", "NW"][Math.floor(((azDeg + 22.5) % 360) / 45)] || "N") as string
        );

  return (
    <div className="obs-panel p-4 sm:p-5 space-y-4 text-xs">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <Sliders className="w-4 h-4 text-sky-400" /> {t("tilt.title")}
          </h3>
          <p className="text-[11px] text-[#8E95A2] mt-0.5">{t("tilt.subtitle")}</p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setShowLiveCam((v) => !v)}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition ${
              showLiveCam
                ? "bg-sky-500/20 border-sky-400/50 text-sky-200"
                : "obs-btn-secondary"
            }`}
          >
            <Eye className="w-3.5 h-3.5 text-sky-400" />
            {showLiveCam ? t("tilt.hideCamPreview") : t("tilt.liveCamPreview")}
          </button>

          <label className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#0F1116] border border-white/[0.1] cursor-pointer">
            <input
              type="checkbox"
              checked={Boolean(value.tilt_correction_enabled)}
              onChange={(e) => updateField({ tilt_correction_enabled: e.target.checked })}
              className="accent-sky-400"
            />
            <span className="font-medium text-white">{t("tilt.enableToggle")}</span>
          </label>

          {onApply && (
            <button
              type="button"
              disabled={applying}
              onClick={onApply}
              className="obs-btn-primary inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs disabled:opacity-50"
            >
              <Check className="w-3.5 h-3.5" /> {t("tilt.applyBtn")}
            </button>
          )}
        </div>
      </div>

      {/* Main 2D Interactive Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-stretch">
        {/* 2D Side-Profile Camera Tilt SVG */}
        <div className="lg:col-span-8 obs-inset p-3 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-1.5 px-1">
            <span className="text-[11px] font-medium text-[#C5CBD5]">{t("tilt.sideViewLabel")}</span>
            <span className="text-[11px] font-mono text-sky-300">{t("tilt.dragHint")}</span>
          </div>

          <svg
            ref={sideSvgRef}
            viewBox="0 0 560 270"
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId);
              setDraggingSide(true);
              handleSidePointer(e);
            }}
            onPointerMove={(e) => {
              if (draggingSide) handleSidePointer(e);
            }}
            onPointerUp={(e) => {
              setDraggingSide(false);
              if (e.currentTarget.hasPointerCapture(e.pointerId)) {
                e.currentTarget.releasePointerCapture(e.pointerId);
              }
            }}
            className={`w-full ${compact ? "h-[205px]" : "h-[240px]"} select-none touch-none rounded-lg bg-[#080A0F] cursor-grab active:cursor-grabbing`}
          >
            <defs>
              <linearGradient id="skyConeGrad" x1="0%" y1="100%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#38BDF8" stopOpacity="0.34" />
                <stop offset="100%" stopColor="#38BDF8" stopOpacity="0.09" />
              </linearGradient>
              <linearGradient id="skyBgGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#0E1726" />
                <stop offset="100%" stopColor="#090B10" />
              </linearGradient>
            </defs>

            {/* Sky background */}
            <rect x="0" y="0" width="560" height={groundY} fill="url(#skyBgGrad)" />

            {/* Ground / rooftop base */}
            <rect x="0" y={groundY} width="560" height={270 - groundY} fill="#12161F" />
            <line x1="16" y1={groundY} x2="546" y2={groundY} stroke="#334155" strokeWidth="1.5" />

            {/* Cloud Deck Layer at ceilingY */}
            <line x1="20" y1={ceilingY} x2="544" y2={ceilingY} stroke="#475569" strokeWidth="1" strokeDasharray="4 4" />
            <g opacity="0.55" fill="#CBD5E1">
              <path d="M 55 58 Q 68 42 86 48 Q 100 34 118 48 Q 134 44 140 58 Z" />
              <path d="M 185 58 Q 204 38 228 45 Q 246 30 268 46 Q 288 42 296 58 Z" />
              <path d="M 332 58 Q 348 40 370 48 Q 388 34 408 46 Q 424 44 432 58 Z" />
              <path d="M 454 58 Q 468 43 488 49 Q 505 37 522 50 Q 535 48 540 58 Z" />
            </g>
            <text x="26" y={ceilingY - 8} fill="#94A3B8" fontSize="10" fontFamily="monospace">
              Cloud Layer (~{cloudAltM} m)
            </text>

            {/* Zenith & Horizon reference lines */}
            <line x1={px} y1={ceilingY - 12} x2={px} y2={py} stroke="#38BDF8" strokeOpacity="0.35" strokeDasharray="3 4" strokeWidth="1" />
            <line x1={px} y1={py} x2="544" y2={py} stroke="#64748B" strokeOpacity="0.35" strokeDasharray="3 4" strokeWidth="1" />
            <text x={px - 26} y={ceilingY + 14} fill="#7DD3FC" fontSize="9.5" fontFamily="monospace">
              Zenith 90°
            </text>
            <text x="488" y={py - 6} fill="#94A3B8" fontSize="9.5" fontFamily="monospace">
              Horizon 0°
            </text>

            {/* Below-horizon warning wedge */}
            {geom.belowPolyStr && (
              <polygon
                points={geom.belowPolyStr}
                fill="#F43F5E"
                fillOpacity="0.22"
                stroke="#FB7185"
                strokeWidth="1"
                strokeDasharray="3 2"
              />
            )}

            {/* Sky FOV cone */}
            <polygon
              points={geom.skyPolyStr}
              fill="url(#skyConeGrad)"
              stroke="#38BDF8"
              strokeOpacity="0.65"
              strokeWidth="1.25"
            />

            {/* Optical Axis Ray + Intercept dot */}
            <line
              x1={px}
              y1={py}
              x2={geom.axisEnd[0]}
              y2={geom.axisEnd[1]}
              stroke="#38BDF8"
              strokeWidth="2"
              strokeDasharray="6 3"
            />
            <circle cx={geom.axisEnd[0]} cy={geom.axisEnd[1]} r="4" fill="#38BDF8" stroke="#080A0F" strokeWidth="1.5" />

            {/* Zenith-Tilt & Horizon-Elevation Arcs */}
            {tiltDeg > 1.5 && (
              <path
                d={`M ${geom.zStartX} ${geom.zStartY} A 54 54 0 0 1 ${geom.axArcX.toFixed(1)} ${geom.axArcY.toFixed(1)}`}
                fill="none"
                stroke="#38BDF8"
                strokeWidth="1.8"
              />
            )}
            <path
              d={`M ${geom.hStartX} ${geom.hStartY} A 72 72 0 0 0 ${geom.elEndX.toFixed(1)} ${geom.elEndY.toFixed(1)}`}
              fill="none"
              stroke="#94A3B8"
              strokeOpacity="0.7"
              strokeWidth="1.3"
            />
            <text x={px + 8} y={py - 62} fill="#38BDF8" fontSize="11" fontWeight="600" fontFamily="monospace">
              Tilt {tiltDeg.toFixed(0)}°
            </text>
            <text x={px + 78} y={py - 10} fill="#CBD5E1" fontSize="10" fontFamily="monospace">
              El {elDeg.toFixed(0)}°
            </text>

            {/* Tripod stand */}
            <line x1={px} y1={py + 4} x2={px - 14} y2={groundY + 12} stroke="#475569" strokeWidth="2.5" strokeLinecap="round" />
            <line x1={px} y1={py + 4} x2={px + 14} y2={groundY + 12} stroke="#475569" strokeWidth="2.5" strokeLinecap="round" />
            <rect x={px - 20} y={groundY + 11} width="40" height="4" rx="2" fill="#334155" />

            {/* Realistic 2D Camera / Phone Silhouette rotating clockwise from Zenith (0° tilt = straight up) */}
            <g transform={`translate(${px}, ${py}) rotate(${tiltDeg})`}>
              {/* Chassis */}
              <rect x="-15" y="-5" width="30" height="14" rx="3.5" fill="#1E293B" stroke="#94A3B8" strokeWidth="1.5" />
              {/* Phone / Camera Screen Side Indicator */}
              <rect x="-11" y="4" width="22" height="3" rx="1" fill="#38BDF8" fillOpacity="0.35" />
              {/* Optical Lens Housing pointing UP (-Y) when tilt=0 */}
              <rect x="-7" y="-13" width="14" height="9" rx="2" fill="#0F172A" stroke="#CBD5E1" strokeWidth="1.3" />
              <line x1="-5" y1="-13" x2="5" y2="-13" stroke="#38BDF8" strokeWidth="2.2" strokeLinecap="round" />
              {/* Pivot Pin */}
              <circle cx="0" cy="2" r="3" fill="#38BDF8" />
            </g>

            {/* Draggable Handle */}
            <circle
              cx={geom.handleX}
              cy={geom.handleY}
              r="9"
              fill="#38BDF8"
              fillOpacity="0.22"
              stroke="#38BDF8"
              strokeWidth="2"
            />
            <circle cx={geom.handleX} cy={geom.handleY} r="3.5" fill="#FFFFFF" />

            {/* Sun elevation indicator on 2D side profile when above horizon */}
            {solar && typeof solar.elevation_deg === "number" && solar.elevation_deg > 0 && (
              <g
                transform={`translate(${
                  px + Math.cos((solar.elevation_deg * Math.PI) / 180) * 145
                }, ${py - Math.sin((solar.elevation_deg * Math.PI) / 180) * 145})`}
              >
                <circle cx="0" cy="0" r="10" fill="#FBBF24" fillOpacity="0.22" />
                <circle cx="0" cy="0" r="5" fill="#FBBF24" stroke="#080A0F" strokeWidth="1.2" />
                <text x="9" y="3" fill="#FDE68A" fontSize="9.5" fontFamily="monospace">
                  ☀ {solar.elevation_deg.toFixed(0)}°
                </text>
              </g>
            )}
          </svg>

          {/* Live Geometric Telemetry Readout */}
          <div className="flex flex-wrap items-center gap-2 pt-2 text-[11px] font-mono tabular-nums">
            <span className="px-2 py-0.5 rounded bg-sky-500/15 text-sky-300 border border-sky-400/25">
              {elDeg >= 88.5
                ? t("tilt.overhead")
                : t("tilt.aimDistance", { km: geom.centerDistKm.toFixed(1) })}
            </span>
            <span className="px-2 py-0.5 rounded bg-white/[0.05] text-[#C5CBD5] border border-white/[0.08]">
              {t("tilt.footprintRange", {
                near: geom.nearDistKm.toFixed(1),
                far: geom.farDistKm.toFixed(1),
              })}
            </span>
            {solar && typeof solar.elevation_deg === "number" && (
              <span className="px-2 py-0.5 rounded bg-amber-500/15 text-amber-300 border border-amber-400/25 inline-flex items-center gap-1">
                <Sun className="w-3 h-3" />
                {t("tilt.sunPos", {
                  el: solar.elevation_deg.toFixed(0),
                  az: Number(solar.azimuth_deg ?? 0).toFixed(0),
                })}
                {solar.in_camera_fov ? ` · ${t("tilt.sunInFov")}` : ""}
              </span>
            )}
            {geom.belowHorizonPct > 0 && (
              <span className="px-2 py-0.5 rounded bg-rose-500/15 text-rose-300 border border-rose-400/25">
                {t("tilt.belowHorizonWarn", { pct: geom.belowHorizonPct })}
              </span>
            )}
          </div>
        </div>

        {/* 2D Top-Down Compass Azimuth SVG */}
        <div className="lg:col-span-4 obs-inset p-3 flex flex-col items-center justify-between">
          <div className="w-full flex items-center justify-between mb-1 px-1">
            <span className="text-[11px] font-medium text-[#C5CBD5] flex items-center gap-1">
              <Compass className="w-3.5 h-3.5 text-sky-400" /> {t("tilt.compassViewLabel")}
            </span>
            <span className="font-mono text-xs font-semibold text-white">
              {azDeg.toFixed(0)}° {cardLabel}
            </span>
          </div>

          <svg
            ref={compassSvgRef}
            viewBox="0 0 180 180"
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId);
              setDraggingCompass(true);
              handleCompassPointer(e);
            }}
            onPointerMove={(e) => {
              if (draggingCompass) handleCompassPointer(e);
            }}
            onPointerUp={(e) => {
              setDraggingCompass(false);
              if (e.currentTarget.hasPointerCapture(e.pointerId)) {
                e.currentTarget.releasePointerCapture(e.pointerId);
              }
            }}
            className="w-[164px] h-[164px] select-none touch-none cursor-grab active:cursor-grabbing"
          >
            <circle cx="90" cy="90" r="70" fill="#080A0F" stroke="#334155" strokeWidth="1.2" />
            <circle cx="90" cy="90" r="44" fill="none" stroke="#1E293B" strokeDasharray="3 3" strokeWidth="1" />
            <line x1="90" y1="20" x2="90" y2="160" stroke="#1E293B" strokeWidth="1" />
            <line x1="20" y1="90" x2="160" y2="90" stroke="#1E293B" strokeWidth="1" />

            {/* Top-down sky FOV projection wedge */}
            <polygon
              points={compassWedge.poly}
              fill="#38BDF8"
              fillOpacity="0.22"
              stroke="#38BDF8"
              strokeWidth="1.4"
            />
            <line
              x1="90"
              y1="90"
              x2={compassWedge.dirEnd[0]}
              y2={compassWedge.dirEnd[1]}
              stroke="#38BDF8"
              strokeWidth="1.8"
              strokeDasharray="3 3"
            />
            <circle cx={compassWedge.aim[0]} cy={compassWedge.aim[1]} r="4" fill="#38BDF8" />
            <circle cx="90" cy="90" r="5" fill="#FFFFFF" stroke="#38BDF8" strokeWidth="2" />

            {/* Sun azimuth marker on compass ring */}
            {solar && typeof solar.azimuth_deg === "number" && (
              <g
                transform={`translate(${
                  90 + Math.sin((solar.azimuth_deg * Math.PI) / 180) * 58
                }, ${90 - Math.cos((solar.azimuth_deg * Math.PI) / 180) * 58})`}
              >
                <circle cx="0" cy="0" r="6.5" fill="#FBBF24" fillOpacity="0.28" />
                <circle cx="0" cy="0" r="3.8" fill="#FBBF24" stroke="#080A0F" strokeWidth="1" />
              </g>
            )}

            {/* Cardinals */}
            <text x="90" y="13" textAnchor="middle" fill="#38BDF8" fontSize="11" fontWeight="700">
              N
            </text>
            <text x="169" y="94" textAnchor="middle" fill="#94A3B8" fontSize="10" fontWeight="600">
              E
            </text>
            <text x="90" y="175" textAnchor="middle" fill="#94A3B8" fontSize="10" fontWeight="600">
              S
            </text>
            <text x="11" y="94" textAnchor="middle" fill="#94A3B8" fontSize="10" fontWeight="600">
              {tCardinal("W")}
            </text>
          </svg>

          {/* Compass quick buttons */}
          <div className="grid grid-cols-4 gap-1 w-full pt-1">
            {CARDINAL_ANGLES.map((c) => {
              const active = Math.abs(((azDeg - c.deg + 540) % 360) - 180) <= 10;
              return (
                <button
                  key={c.deg}
                  type="button"
                  onClick={() => updateField({ orientation_azimuth_deg: c.deg })}
                  className={`py-1 rounded text-[11px] font-mono border transition ${
                    active
                      ? "bg-sky-500/20 border-sky-400/50 text-sky-200"
                      : "bg-[#0F1116] border-white/[0.08] text-[#9AA1AD] hover:text-white"
                  }`}
                >
                  {c.deg === 225 ? tCardinal("SW") : c.deg === 270 ? tCardinal("W") : c.deg === 315 ? tCardinal("NW") : c.label}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Sliders, Quick Presets & Optical Inputs */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        <div className="obs-inset p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[#C5CBD5]">{t("tilt.tiltFromVertical")}</span>
            <span className="font-mono font-bold text-sky-300 tabular-nums">
              {tiltDeg.toFixed(0)}° <span className="text-[#646B78] font-normal">(El {elDeg.toFixed(0)}°)</span>
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={110}
            step={1}
            value={Math.round(tiltDeg)}
            onChange={(e) => updateField({ orientation_elevation_deg: clamp(90 - Number(e.target.value), -20, 90) })}
            className="w-full accent-sky-400"
          />
          <div className="flex flex-wrap gap-1 pt-0.5">
            {[0, 15, 30, 45, 60, 75, 90, 100].map((presetTilt) => {
              const active = Math.abs(tiltDeg - presetTilt) <= 1;
              return (
                <button
                  key={presetTilt}
                  type="button"
                  onClick={() => updateField({ orientation_elevation_deg: 90 - presetTilt })}
                  className={`px-2 py-0.5 rounded text-[10px] font-mono border ${
                    active
                      ? "bg-sky-500/20 border-sky-400/50 text-sky-200"
                      : "bg-[#0B0C0E] border-white/[0.08] text-[#9AA1AD] hover:text-white"
                  }`}
                >
                  {presetTilt}°
                </button>
              );
            })}
          </div>
        </div>

        <div className="obs-inset p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[#C5CBD5]">{t("tilt.azimuthLabel")}</span>
            <span className="font-mono font-bold text-white tabular-nums">
              {azDeg.toFixed(0)}° {cardLabel}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={359}
            step={1}
            value={Math.round(azDeg)}
            onChange={(e) => updateField({ orientation_azimuth_deg: Number(e.target.value) })}
            className="w-full accent-sky-400"
          />
          <div className="flex items-center justify-between text-[11px] text-[#8E95A2]">
            <span>{t("tilt.rollLabel")}</span>
            <input
              type="number"
              min={-45}
              max={45}
              step={1}
              value={rollDeg}
              onChange={(e) => updateField({ orientation_roll_deg: clamp(Number(e.target.value) || 0, -45, 45) })}
              className="obs-input w-20 py-0.5 text-right font-mono tabular-nums"
            />
          </div>
        </div>

        <div className="obs-inset p-3 grid grid-cols-3 gap-2">
          <label className="space-y-1">
            <span className="text-[#8E95A2] text-[10px]">{t("tilt.fovH")}</span>
            <input
              type="number"
              min={10}
              max={170}
              value={value.fov_horizontal_deg ?? 72}
              onChange={(e) => updateField({ fov_horizontal_deg: Number(e.target.value) })}
              onBlur={() => updateField({ fov_horizontal_deg: clamp(Number(value.fov_horizontal_deg) || 72, 10, 170) })}
              className="obs-input w-full font-mono tabular-nums"
            />
          </label>
          <label className="space-y-1">
            <span className="text-[#8E95A2] text-[10px]">{t("tilt.fovV")}</span>
            <input
              type="number"
              min={10}
              max={170}
              value={value.fov_vertical_deg ?? 45}
              onChange={(e) => updateField({ fov_vertical_deg: Number(e.target.value) })}
              onBlur={() => updateField({ fov_vertical_deg: clamp(Number(value.fov_vertical_deg) || 45, 10, 170) })}
              className="obs-input w-full font-mono tabular-nums"
            />
          </label>
          <label className="space-y-1" title={t("tilt.cloudAltHint")}>
            <div className="flex items-center justify-between gap-1">
              <span className="text-[#8E95A2] text-[10px]">{t("tilt.cloudAltM")}</span>
              {value.assumed_cloud_altitude_m !== null ? (
                <button
                  type="button"
                  onClick={() => updateField({ assumed_cloud_altitude_m: null })}
                  className="text-[9px] font-mono text-sky-300 hover:underline"
                >
                  {t("tilt.autoLclBtn")}
                </button>
              ) : (
                <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-400/25">
                  AUTO
                </span>
              )}
            </div>
            <input
              type="number"
              min={100}
              max={15000}
              step={100}
              placeholder={`Auto (~${Math.round(effectiveCloudAltitudeM || 2000)}m)`}
              value={value.assumed_cloud_altitude_m ?? ""}
              onChange={(e) => {
                const raw = e.target.value.trim();
                updateField({
                  assumed_cloud_altitude_m: raw === "" ? null : Number(raw),
                });
              }}
              onBlur={() => {
                if (value.assumed_cloud_altitude_m !== null) {
                  updateField({
                    assumed_cloud_altitude_m: clamp(Number(value.assumed_cloud_altitude_m) || 2000, 100, 15000),
                  });
                }
              }}
              className="obs-input w-full font-mono tabular-nums"
            />
          </label>
        </div>
      </div>

      {/* Live Camera Photo with 3D Horizon, Elevation Iso-Curves, Compass & Sun Overlay */}
      {showLiveCam && (
        <div className="obs-inset p-3 space-y-2">
          <div className="flex items-center justify-between text-[11px]">
            <span className="font-medium text-white flex items-center gap-1.5">
              <Eye className="w-3.5 h-3.5 text-sky-400" /> {t("tilt.liveCamPreview")}
            </span>
            <span className="font-mono text-[#8E95A2]">
              Tilt {tiltDeg.toFixed(0)}° (El {elDeg.toFixed(0)}°) · Az {azDeg.toFixed(0)}° {cardLabel} · FOV {fovH}°×{fovV}°
            </span>
          </div>
          <div className="relative aspect-video max-h-[280px] mx-auto bg-[#08090B] rounded-lg overflow-hidden border border-white/[0.08]">
            <img
              src="/api/camera/latest?layer=raw"
              alt="Live camera calibration feed"
              className="w-full h-full object-contain"
              onError={(e) => {
                (e.currentTarget as HTMLImageElement).style.opacity = "0.15";
              }}
            />
            <SkyHorizonOverlay
              elevationDeg={elDeg}
              azimuthDeg={azDeg}
              rollDeg={rollDeg}
              fovHorizontalDeg={fovH}
              fovVerticalDeg={fovV}
              solar={solar}
            />
          </div>
        </div>
      )}
    </div>
  );
};
