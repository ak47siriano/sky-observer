import React, { useMemo, useRef, useState } from "react";
import { Compass, Check, Eye, Link2, Unlink2, Sliders, Sparkles, Sun, Maximize2 } from "lucide-react";
import { useI18n } from "../i18n/I18nContext";
import { api } from "../services/api";
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

const CARDINAL_ANGLES: { key: "N" | "NE" | "E" | "SE" | "S" | "SW" | "W" | "NW"; deg: number }[] = [
  { key: "N", deg: 0 },
  { key: "NE", deg: 45 },
  { key: "E", deg: 90 },
  { key: "SE", deg: 135 },
  { key: "S", deg: 180 },
  { key: "SW", deg: 225 },
  { key: "W", deg: 270 },
  { key: "NW", deg: 315 },
];

const LENS_PRESETS: { label: string; fovH: number; fovV: number }[] = [
  { label: "Tele 45°", fovH: 45, fovV: 27 },
  { label: "Phone 72°", fovH: 72, fovV: 45 },
  { label: "Wide 95°", fovH: 95, fovV: 62 },
  { label: "Action 120°", fovH: 120, fovV: 84 },
  { label: "Fisheye 155°", fovH: 155, fovV: 118 },
];

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const rad = (deg: number) => (deg * Math.PI) / 180;
const degFromRad = (r: number) => (r * 180) / Math.PI;

/** Derive matching 16:9 vertical FOV from horizontal FOV. */
function derive16x9VerticalFov(fovH: number): number {
  const halfH = rad(clamp(fovH, 10, 170) / 2);
  const halfV = Math.atan(Math.tan(halfH) * (9 / 16));
  return Math.round(clamp(degFromRad(halfV * 2), 10, 150));
}

/** Build an SVG annular sector (ring wedge) between innerR and outerR across [startDeg, endDeg] clockwise from North (0° = up). */
function buildCompassSectorPath(
  cx: number,
  cy: number,
  innerR: number,
  outerR: number,
  centerAzDeg: number,
  spanDeg: number
): string {
  const half = clamp(spanDeg, 2, 358) / 2;
  const a1 = rad(centerAzDeg - half);
  const a2 = rad(centerAzDeg + half);
  const largeArc = spanDeg > 180 ? 1 : 0;

  const ox1 = cx + Math.sin(a1) * outerR;
  const oy1 = cy - Math.cos(a1) * outerR;
  const ox2 = cx + Math.sin(a2) * outerR;
  const oy2 = cy - Math.cos(a2) * outerR;

  if (innerR <= 0.5) {
    return `M ${cx.toFixed(1)} ${cy.toFixed(1)} L ${ox1.toFixed(1)} ${oy1.toFixed(1)} A ${outerR.toFixed(1)} ${outerR.toFixed(1)} 0 ${largeArc} 1 ${ox2.toFixed(1)} ${oy2.toFixed(1)} Z`;
  }
  const ix1 = cx + Math.sin(a1) * innerR;
  const iy1 = cy - Math.cos(a1) * innerR;
  const ix2 = cx + Math.sin(a2) * innerR;
  const iy2 = cy - Math.cos(a2) * innerR;
  return [
    `M ${ix1.toFixed(1)} ${iy1.toFixed(1)}`,
    `L ${ox1.toFixed(1)} ${oy1.toFixed(1)}`,
    `A ${outerR.toFixed(1)} ${outerR.toFixed(1)} 0 ${largeArc} 1 ${ox2.toFixed(1)} ${oy2.toFixed(1)}`,
    `L ${ix2.toFixed(1)} ${iy2.toFixed(1)}`,
    `A ${innerR.toFixed(1)} ${innerR.toFixed(1)} 0 ${largeArc} 0 ${ix1.toFixed(1)} ${iy1.toFixed(1)}`,
    "Z",
  ].join(" ");
}

function buildCompassArcPath(cx: number, cy: number, r: number, centerAzDeg: number, spanDeg: number): string {
  const half = clamp(spanDeg, 2, 358) / 2;
  const a1 = rad(centerAzDeg - half);
  const a2 = rad(centerAzDeg + half);
  const largeArc = spanDeg > 180 ? 1 : 0;
  const x1 = cx + Math.sin(a1) * r;
  const y1 = cy - Math.cos(a1) * r;
  const x2 = cx + Math.sin(a2) * r;
  const y2 = cy - Math.cos(a2) * r;
  return `M ${x1.toFixed(1)} ${y1.toFixed(1)} A ${r.toFixed(1)} ${r.toFixed(1)} 0 ${largeArc} 1 ${x2.toFixed(1)} ${y2.toFixed(1)}`;
}

export const CameraTiltCalibrator: React.FC<Props> = ({
  value,
  onChange,
  onApply,
  applying = false,
  compact = false,
  solar = null,
  effectiveCloudAltitudeM,
}) => {
  const { t, tCardinal, locale } = useI18n();
  const isIt = locale === "it";
  const sideSvgRef = useRef<SVGSVGElement | null>(null);
  const compassSvgRef = useRef<SVGSVGElement | null>(null);
  const [draggingSide, setDraggingSide] = useState(false);
  const [draggingCompass, setDraggingCompass] = useState(false);
  const [showLiveCam, setShowLiveCam] = useState(!compact);
  const [lockAspect16x9, setLockAspect16x9] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const [detectNote, setDetectNote] = useState<{ text: string; tone: "ok" | "warn" | "err" } | null>(null);
  const [camPreviewRetry, setCamPreviewRetry] = useState(0);

  const elDeg = clamp(Number(value.orientation_elevation_deg ?? 65), -20, 90);
  const tiltDeg = Math.round((90 - elDeg) * 10) / 10;
  const azDeg = ((Number(value.orientation_azimuth_deg ?? 0) % 360) + 360) % 360;
  const rollDeg = clamp(Number(value.orientation_roll_deg ?? 0), -45, 45);
  const fovH = clamp(Number(value.fov_horizontal_deg || 72), 10, 170);
  const fovV = clamp(Number(value.fov_vertical_deg || 45), 10, 150);
  const cloudAltM = clamp(Number(value.assumed_cloud_altitude_m || effectiveCloudAltitudeM || 2000), 100, 15000);

  const updateField = (patch: Partial<CameraGeometryValues>) => {
    onChange({
      ...value,
      ...patch,
    });
  };

  const setFovH = (nextH: number) => {
    const cH = clamp(Math.round(nextH), 10, 170);
    if (lockAspect16x9) {
      updateField({ fov_horizontal_deg: cH, fov_vertical_deg: derive16x9VerticalFov(cH) });
    } else {
      updateField({ fov_horizontal_deg: cH });
    }
  };

  const handleAutoDetectHorizon = async () => {
    setDetecting(true);
    setDetectNote(null);
    try {
      const res = await api.autoDetectCameraTilt({
        apply: false,
        fov_horizontal_deg: fovH,
        fov_vertical_deg: fovV,
        recalculate_latest: false,
      });
      updateField({
        orientation_elevation_deg: clamp(Number(res.orientation_elevation_deg ?? elDeg), -20, 90),
        orientation_roll_deg: clamp(Number(res.orientation_roll_deg ?? 0), -45, 45),
        tilt_correction_enabled: true,
      });
      setShowLiveCam(true);
      const confPct = Math.round(Number(res.confidence ?? 0.5) * 100);
      if (res.detected_horizon) {
        setDetectNote({
          tone: "ok",
          text: t("tilt.detectedHorizon", {
            el: Number(res.orientation_elevation_deg ?? 0).toFixed(1),
            roll: Number(res.orientation_roll_deg ?? 0).toFixed(1),
            conf: confPct,
          }),
        });
      } else {
        setDetectNote({
          tone: "warn",
          text: t("tilt.detectedZenith", {
            el: Number(res.orientation_elevation_deg ?? 85).toFixed(0),
            conf: confPct,
          }),
        });
      }
    } catch (e: unknown) {
      setDetectNote({
        tone: "err",
        text: e instanceof Error ? e.message : t("tilt.detectFailed"),
      });
    } finally {
      setDetecting(false);
    }
  };

  // =========================================================================
  // 1. SIDE-PROFILE 2D ELEVATION GEOMETRY (viewBox 560 x 250 — full-bleed)
  // =========================================================================
  const sideW = 560;
  const sideH = 250;
  const px = 98;
  const py = 196;
  const ceilingY = 44;
  const groundY = 212;

  const sideGeom = useMemo(() => {
    const elRad = rad(elDeg);
    const halfV = fovV / 2;
    const upperAngle = elDeg + halfV;
    const lowerAngle = elDeg - halfV;

    const traceRay = (angleDeg: number, capR = 440): [number, number] => {
      const a = rad(angleDeg);
      const vx = Math.cos(a);
      const vy = -Math.sin(a);
      let tHit = capR;
      if (vy < -0.015) {
        const tCeil = (ceilingY - py) / vy;
        if (tCeil > 0) tHit = Math.min(tHit, tCeil);
      } else if (vy > 0.015) {
        const tGround = (sideH - 8 - py) / vy;
        if (tGround > 0) tHit = Math.min(tHit, tGround);
      }
      if (vx > 0.015) {
        tHit = Math.min(tHit, (sideW - 14 - px) / vx);
      } else if (vx < -0.015) {
        tHit = Math.min(tHit, (14 - px) / vx);
      }
      return [px + vx * tHit, py + vy * tHit];
    };

    const sampleWedge = (startDeg: number, endDeg: number, capR = 440): string => {
      if (endDeg <= startDeg + 0.1) return "";
      const pts: [number, number][] = [[px, py]];
      const steps = 20;
      for (let i = 0; i <= steps; i++) {
        const ang = startDeg + ((endDeg - startDeg) * i) / steps;
        pts.push(traceRay(ang, capR));
      }
      return pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
    };

    // Split cone into 3 physical regimes:
    // (a) Usable projected sky: >= 15° elevation (zenith <= 75°)
    // (b) Low-horizon strip: 0°..15° elevation (visible sky, clipped from cloud-map projection)
    // (c) Below horizon: < 0° elevation (auto-excluded ground/buildings)
    const usableStart = Math.max(15, lowerAngle);
    const usableEnd = Math.max(usableStart, Math.min(175, upperAngle));
    const usablePoly = sampleWedge(usableStart, usableEnd, 440);

    const lowStart = Math.max(0, lowerAngle);
    const lowEnd = Math.min(15, upperAngle);
    const lowHorizonPoly = sampleWedge(lowStart, lowEnd, 440);

    const belowStart = Math.max(-42, lowerAngle);
    const belowEnd = Math.min(0, upperAngle);
    const belowPoly = sampleWedge(belowStart, belowEnd, 240);

    // Full vertical FOV boundary rays (for crisp visual feedback when changing fovV)
    const upperRayEnd = traceRay(Math.min(175, upperAngle), 440);
    const lowerRayEnd = traceRay(Math.max(-42, lowerAngle), lowerAngle < 0 ? 240 : 440);

    // Vertical FOV angular arc at radius 76 around camera
    const fovArcR = 76;
    const clampedLoArc = clamp(lowerAngle, -35, 165);
    const clampedHiArc = clamp(upperAngle, clampedLoArc + 2, 175);
    const f1x = px + Math.cos(rad(clampedLoArc)) * fovArcR;
    const f1y = py - Math.sin(rad(clampedLoArc)) * fovArcR;
    const f2x = px + Math.cos(rad(clampedHiArc)) * fovArcR;
    const f2y = py - Math.sin(rad(clampedHiArc)) * fovArcR;
    const fovArcLarge = clampedHiArc - clampedLoArc > 180 ? 1 : 0;
    const fovArcPath = `M ${f1x.toFixed(1)} ${f1y.toFixed(1)} A ${fovArcR} ${fovArcR} 0 ${fovArcLarge} 0 ${f2x.toFixed(1)} ${f2y.toFixed(1)}`;

    // 15° projection cutoff reference line
    const cutoff15End = traceRay(15, 440);

    // Optical axis & draggable handle
    const axisEnd = traceRay(elDeg, 440);
    const handleDist = 128;
    const handleX = clamp(px + Math.cos(elRad) * handleDist, 24, sideW - 24);
    const handleY = clamp(py - Math.sin(elRad) * handleDist, 26, sideH - 22);

    // Tilt arc from zenith (90°) to optical axis
    const arcR = 46;
    const zStartX = px;
    const zStartY = py - arcR;
    const axArcX = px + Math.cos(elRad) * arcR;
    const axArcY = py - Math.sin(elRad) * arcR;
    const tiltArcLarge = tiltDeg > 180 ? 1 : 0;

    // Physical ground distances at cloudAltM
    const altKm = cloudAltM / 1000;
    const centerDistKm = elDeg >= 88.5 ? 0 : Math.min(28, altKm / Math.tan(rad(Math.max(2, elDeg))));
    const nearDistKm = upperAngle >= 90 ? 0 : Math.min(28, altKm / Math.tan(rad(Math.max(4, upperAngle))));
    const farDistKm = Math.min(28, altKm / Math.tan(rad(Math.max(15, lowerAngle))));
    const belowHorizonPct = lowerAngle < 0 ? Math.min(100, Math.round((-lowerAngle / fovV) * 100)) : 0;

    return {
      usablePoly,
      lowHorizonPoly,
      belowPoly,
      upperRayEnd,
      lowerRayEnd,
      fovArcPath,
      cutoff15End,
      axisEnd,
      handleX,
      handleY,
      zStartX,
      zStartY,
      axArcX,
      axArcY,
      tiltArcLarge,
      centerDistKm,
      nearDistKm: Math.min(nearDistKm, farDistKm),
      farDistKm: Math.max(nearDistKm, farDistKm),
      belowHorizonPct,
    };
  }, [cloudAltM, elDeg, fovV, tiltDeg]);

  // =========================================================================
  // 2. TOP-DOWN PLAN VIEW: AZIMUTH & TRUE HORIZONTAL FOV (viewBox 340 x 250)
  // =========================================================================
  const topW = 340;
  const topH = 250;
  const cx = 170;
  const cy = 128;
  const maxR = 96;

  const topGeom = useMemo(() => {
    const azRad = rad(azDeg);
    const halfH = fovH / 2;
    const leftRad = rad(azDeg - halfH);
    const rightRad = rad(azDeg + halfH);

    // Fixed physical scale so changing FOV H NEVER rescales the dial!
    // Map range dynamically picks a clean round ceiling based on cloudAltM (e.g. 10 km or 20 km).
    const maxRangeKm = cloudAltM >= 3500 ? 28 : cloudAltM >= 1800 ? 16 : 10;
    const kmToR = (km: number) => clamp((km / maxRangeKm) * maxR, 0, maxR);

    const nearR = kmToR(sideGeom.nearDistKm);
    const farR = Math.max(nearR + 14, kmToR(sideGeom.farDistKm));
    const centerR = kmToR(sideGeom.centerDistKm);

    // Full camera horizontal FOV sector (always exact geometric angle fovH!)
    const fullSectorPath = buildCompassSectorPath(cx, cy, 0, maxR, azDeg, fovH);
    // Projected sky-cloud footprint annulus inside the horizontal FOV
    const footprintPath = buildCompassSectorPath(cx, cy, nearR, Math.min(maxR, farR), azDeg, fovH);
    // Crisp outer angular-span arc highlighting FOV H
    const fovSpanArc = buildCompassArcPath(cx, cy, maxR + 5, azDeg, fovH);

    const leftEdge: [number, number] = [cx + Math.sin(leftRad) * maxR, cy - Math.cos(leftRad) * maxR];
    const rightEdge: [number, number] = [cx + Math.sin(rightRad) * maxR, cy - Math.cos(rightRad) * maxR];
    const axisEnd: [number, number] = [cx + Math.sin(azRad) * maxR, cy - Math.cos(azRad) * maxR];
    const aimPt: [number, number] = [cx + Math.sin(azRad) * centerR, cy - Math.cos(azRad) * centerR];
    const handlePt: [number, number] = [cx + Math.sin(azRad) * (maxR * 0.78), cy - Math.cos(azRad) * (maxR * 0.78)];

    // Horizontal arc width in km at the optical center distance
    const widthAtCenterKm =
      sideGeom.centerDistKm > 0.2
        ? 2 * sideGeom.centerDistKm * Math.tan(rad(Math.min(82, halfH)))
        : 2 * (cloudAltM / 1000) * Math.tan(rad(Math.min(82, halfH)));

    return {
      maxRangeKm,
      fullSectorPath,
      footprintPath,
      fovSpanArc,
      leftEdge,
      rightEdge,
      axisEnd,
      aimPt,
      handlePt,
      widthAtCenterKm: Math.min(99, widthAtCenterKm),
    };
  }, [azDeg, cloudAltM, fovH, sideGeom.centerDistKm, sideGeom.farDistKm, sideGeom.nearDistKm]);

  const handleSidePointer = (e: React.PointerEvent<SVGSVGElement>) => {
    const svg = sideSvgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * sideW;
    const y = ((e.clientY - rect.top) / rect.height) * sideH;
    const dx = Math.max(0.01, x - px);
    const dy = py - y;
    const angle = Math.round(degFromRad(Math.atan2(dy, dx)));
    updateField({ orientation_elevation_deg: clamp(angle, -20, 90) });
  };

  const handleCompassPointer = (e: React.PointerEvent<SVGSVGElement>) => {
    const svg = compassSvgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * topW;
    const y = ((e.clientY - rect.top) / rect.height) * topH;
    const deg = Math.round((degFromRad(Math.atan2(x - cx, cy - y)) + 360) % 360);
    updateField({ orientation_azimuth_deg: deg });
  };

  const cardinalKey = CARDINAL_ANGLES[Math.floor(((azDeg + 22.5) % 360) / 45)]?.key || "N";
  const cardLabel = tCardinal(cardinalKey);

  return (
    <div className="obs-panel p-4 space-y-3.5 text-xs overflow-hidden">
      {/* Compact Non-Overflowing Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2.5 border-b border-white/[0.06] pb-3">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <Sliders className="w-4 h-4 text-sky-400 shrink-0" />
            <span className="truncate">{t("tilt.title")}</span>
          </h3>
          <p className="text-[11px] text-[#8E95A2] mt-0.5">{t("tilt.subtitle")}</p>
        </div>

        <div className="flex flex-wrap items-center gap-1.5 shrink-0">
          <button
            type="button"
            disabled={detecting}
            onClick={handleAutoDetectHorizon}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-medium bg-emerald-500/15 border border-emerald-400/35 text-emerald-200 hover:bg-emerald-500/25 disabled:opacity-50 transition"
          >
            <Sparkles className={`w-3.5 h-3.5 text-emerald-400 ${detecting ? "animate-spin" : ""}`} />
            {detecting ? t("tilt.detectingBtn") : t("tilt.autoDetectBtn")}
          </button>

          <button
            type="button"
            onClick={() => setShowLiveCam((v) => !v)}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-medium border transition ${
              showLiveCam
                ? "bg-sky-500/20 border-sky-400/45 text-sky-200"
                : "bg-[#0F1219] border-white/[0.1] text-[#C5CBD5] hover:text-white"
            }`}
          >
            <Eye className="w-3.5 h-3.5 text-sky-400" />
            {showLiveCam ? t("tilt.hideCamPreview") : t("tilt.liveCamPreview")}
          </button>

          <label className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-[#0F1219] border border-white/[0.1] cursor-pointer text-[11px]">
            <input
              type="checkbox"
              checked={Boolean(value.tilt_correction_enabled)}
              onChange={(e) => updateField({ tilt_correction_enabled: e.target.checked })}
              className="accent-sky-400"
            />
            <span className="text-[#E2E8F0]">{t("tilt.enableToggle")}</span>
          </label>

          {onApply && (
            <button
              type="button"
              disabled={applying}
              onClick={onApply}
              className="obs-btn-primary inline-flex items-center gap-1.5 px-3 py-1.5 text-[11px] disabled:opacity-50"
            >
              <Check className="w-3.5 h-3.5" /> {t("tilt.applyBtn")}
            </button>
          )}
        </div>
      </div>

      {detectNote && (
        <div
          className={`px-3 py-1.5 rounded-lg border text-[11px] font-mono ${
            detectNote.tone === "ok"
              ? "bg-emerald-500/12 border-emerald-400/35 text-emerald-200"
              : detectNote.tone === "warn"
              ? "bg-amber-500/12 border-amber-400/35 text-amber-200"
              : "bg-rose-500/12 border-rose-400/35 text-rose-200"
          }`}
        >
          {detectNote.text}
        </div>
      )}

      {/* Dual-Viewport Optical Cockpit: Side Profile (Tilt + Vert FOV) + Top-Down Plan (Azimuth + TRUE Horiz FOV) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 items-stretch">
        {/* LEFT VIEWPORT: Side Elevation Profile (Pitch & Vertical FOV) */}
        <div className="lg:col-span-7 rounded-xl bg-[#0B0E14] border border-white/[0.08] overflow-hidden flex flex-col">
          <div className="flex items-center justify-between px-3 py-2 border-b border-white/[0.06] bg-white/[0.02]">
            <span className="text-[11px] font-medium text-[#D1D8E3]">{t("tilt.sideViewLabel")}</span>
            <div className="flex items-center gap-2 font-mono text-[11px]">
              <span className="text-sky-300 font-semibold">Tilt {tiltDeg.toFixed(0)}°</span>
              <span className="text-[#64748B]">·</span>
              <span className="text-[#CBD5E1]">El {elDeg.toFixed(0)}°</span>
              <span className="text-[#64748B]">·</span>
              <span className="text-indigo-300">FOV V {fovV.toFixed(0)}°</span>
            </div>
          </div>

          <div className="relative w-full flex-1 flex items-center bg-[#090C12]">
            <svg
              ref={sideSvgRef}
              viewBox={`0 0 ${sideW} ${sideH}`}
              preserveAspectRatio="xMidYMid meet"
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
              className="w-full aspect-[56/25] block select-none touch-none cursor-grab active:cursor-grabbing"
            >
              <defs>
                <linearGradient id="calSkyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#101A2C" />
                  <stop offset="100%" stopColor="#090C12" />
                </linearGradient>
                <linearGradient id="calConeUsable" x1="0%" y1="100%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#38BDF8" stopOpacity="0.34" />
                  <stop offset="100%" stopColor="#38BDF8" stopOpacity="0.10" />
                </linearGradient>
              </defs>

              {/* Full-bleed sky & ground planes */}
              <rect x="0" y="0" width={sideW} height={groundY} fill="url(#calSkyGrad)" />
              <rect x="0" y={groundY} width={sideW} height={sideH - groundY} fill="#111620" />

              {/* Subtle technical grid */}
              {[90, 140].map((gy) => (
                <line key={gy} x1="0" y1={gy} x2={sideW} y2={gy} stroke="#1E293B" strokeOpacity="0.45" strokeWidth="1" />
              ))}
              {[200, 310, 420, 520].map((gx) => (
                <line key={gx} x1={gx} y1={ceilingY} x2={gx} y2={groundY} stroke="#1E293B" strokeOpacity="0.4" strokeWidth="1" />
              ))}

              {/* Cloud deck reference line */}
              <line x1="12" y1={ceilingY} x2={sideW - 12} y2={ceilingY} stroke="#64748B" strokeOpacity="0.65" strokeWidth="1" strokeDasharray="5 4" />
              <rect x={sideW - 148} y="16" width="136" height="20" rx="4" fill="#0F172A" stroke="#334155" strokeWidth="0.8" />
              <text x={sideW - 80} y="30" textAnchor="middle" fill="#CBD5E1" fontSize="10" fontFamily="monospace">
                {isIt ? "Quota" : "Ceiling"} ~{Math.round(cloudAltM)} m
              </text>

              {/* Zenith & Horizon reference lines */}
              <line x1={px} y1={ceilingY - 8} x2={px} y2={py} stroke="#38BDF8" strokeOpacity="0.32" strokeDasharray="3 4" strokeWidth="1" />
              <line x1={px} y1={py} x2={sideW - 12} y2={py} stroke="#64748B" strokeOpacity="0.45" strokeDasharray="4 4" strokeWidth="1" />
              <line x1="0" y1={groundY} x2={sideW} y2={groundY} stroke="#334155" strokeWidth="1.2" />

              {/* 15° minimum projection elevation cutoff line */}
              <line
                x1={px}
                y1={py}
                x2={sideGeom.cutoff15End[0]}
                y2={sideGeom.cutoff15End[1]}
                stroke="#F59E0B"
                strokeOpacity="0.32"
                strokeDasharray="3 4"
                strokeWidth="1"
              />

              {/* Clean non-overlapping axis captions */}
              <text x={px} y="24" textAnchor="middle" fill="#7DD3FC" fontSize="9.5" fontFamily="monospace">
                {isIt ? "Zenit 90°" : "Zenith 90°"}
              </text>
              <text x={sideW - 16} y={py - 5} textAnchor="end" fill="#94A3B8" fontSize="9.5" fontFamily="monospace">
                {isIt ? "Orizzonte 0°" : "Horizon 0°"}
              </text>
              <text x={sideW - 16} y={sideGeom.cutoff15End[1] - 4} textAnchor="end" fill="#FBBF24" fillOpacity="0.7" fontSize="9" fontFamily="monospace">
                {isIt ? "15° min proiez." : "15° min proj."}
              </text>

              {/* Cone Region 1: Below horizon (< 0° excluded) */}
              {sideGeom.belowPoly && (
                <polygon
                  points={sideGeom.belowPoly}
                  fill="#F43F5E"
                  fillOpacity="0.22"
                  stroke="#FB7185"
                  strokeOpacity="0.75"
                  strokeWidth="1"
                  strokeDasharray="3 2"
                />
              )}

              {/* Cone Region 2: Low horizon (0°..15° visible sky, clipped on cloud map) */}
              {sideGeom.lowHorizonPoly && (
                <polygon
                  points={sideGeom.lowHorizonPoly}
                  fill="#F59E0B"
                  fillOpacity="0.14"
                  stroke="#F59E0B"
                  strokeOpacity="0.35"
                  strokeWidth="1"
                />
              )}

              {/* Cone Region 3: Usable sky projection (>= 15°) */}
              {sideGeom.usablePoly && (
                <polygon
                  points={sideGeom.usablePoly}
                  fill="url(#calConeUsable)"
                  stroke="#38BDF8"
                  strokeOpacity="0.72"
                  strokeWidth="1.25"
                />
              )}

              {/* Explicit Upper & Lower Vertical FOV Boundary Rays */}
              <line x1={px} y1={py} x2={sideGeom.upperRayEnd[0]} y2={sideGeom.upperRayEnd[1]} stroke="#818CF8" strokeOpacity="0.85" strokeWidth="1.25" />
              <line x1={px} y1={py} x2={sideGeom.lowerRayEnd[0]} y2={sideGeom.lowerRayEnd[1]} stroke="#818CF8" strokeOpacity="0.85" strokeWidth="1.25" />
              <path d={sideGeom.fovArcPath} fill="none" stroke="#818CF8" strokeOpacity="0.75" strokeWidth="1.5" strokeDasharray="3 2" />

              {/* Optical Axis Ray + Cloud Intercept */}
              <line x1={px} y1={py} x2={sideGeom.axisEnd[0]} y2={sideGeom.axisEnd[1]} stroke="#38BDF8" strokeWidth="2" strokeDasharray="6 3" />
              <circle cx={sideGeom.axisEnd[0]} cy={sideGeom.axisEnd[1]} r="3.8" fill="#38BDF8" stroke="#090C12" strokeWidth="1.5" />

              {/* Zenith-to-Axis Tilt Arc */}
              {tiltDeg > 2 && (
                <path
                  d={`M ${sideGeom.zStartX} ${sideGeom.zStartY} A 46 46 0 ${sideGeom.tiltArcLarge} 1 ${sideGeom.axArcX.toFixed(1)} ${sideGeom.axArcY.toFixed(1)}`}
                  fill="none"
                  stroke="#38BDF8"
                  strokeWidth="1.8"
                />
              )}

              {/* Tripod & Camera Silhouette */}
              <line x1={px} y1={py + 4} x2={px - 12} y2={groundY + 12} stroke="#475569" strokeWidth="2.2" strokeLinecap="round" />
              <line x1={px} y1={py + 4} x2={px + 12} y2={groundY + 12} stroke="#475569" strokeWidth="2.2" strokeLinecap="round" />
              <rect x={px - 18} y={groundY + 11} width="36" height="3.5" rx="1.5" fill="#334155" />
              <g transform={`translate(${px}, ${py}) rotate(${tiltDeg})`}>
                <rect x="-13" y="-5" width="26" height="12" rx="3" fill="#1E293B" stroke="#94A3B8" strokeWidth="1.4" />
                <rect x="-6" y="-12" width="12" height="8" rx="1.8" fill="#0F172A" stroke="#CBD5E1" strokeWidth="1.2" />
                <line x1="-4.5" y1="-12" x2="4.5" y2="-12" stroke="#38BDF8" strokeWidth="2" strokeLinecap="round" />
                <circle cx="0" cy="1" r="2.6" fill="#38BDF8" />
              </g>

              {/* Draggable Tilt Handle */}
              <circle cx={sideGeom.handleX} cy={sideGeom.handleY} r="10" fill="#38BDF8" fillOpacity="0.2" stroke="#38BDF8" strokeWidth="1.8" />
              <circle cx={sideGeom.handleX} cy={sideGeom.handleY} r="3.5" fill="#FFFFFF" />

              {/* Sun altitude marker */}
              {solar && typeof solar.elevation_deg === "number" && solar.elevation_deg > 0 && (
                <g
                  transform={`translate(${clamp(px + Math.cos(rad(solar.elevation_deg)) * 150, 30, sideW - 70)}, ${clamp(
                    py - Math.sin(rad(solar.elevation_deg)) * 150,
                    ceilingY + 16,
                    py - 16
                  )})`}
                >
                  <circle cx="0" cy="0" r="8" fill="#FBBF24" fillOpacity="0.22" />
                  <circle cx="0" cy="0" r="4.2" fill="#FBBF24" stroke="#090C12" strokeWidth="1" />
                  <text x="10" y="3.5" fill="#FDE68A" fontSize="9.5" fontFamily="monospace">
                    {solar.elevation_deg.toFixed(0)}°
                  </text>
                </g>
              )}

              {/* Bottom-left drag hint */}
              <text x="14" y={sideH - 8} fill="#64748B" fontSize="9.5" fontFamily="monospace">
                {t("tilt.dragHint")}
              </text>
            </svg>
          </div>
        </div>

        {/* RIGHT VIEWPORT: Top-Down Plan View (Azimuth & TRUE Horizontal FOV Sector) */}
        <div className="lg:col-span-5 rounded-xl bg-[#0B0E14] border border-white/[0.08] overflow-hidden flex flex-col">
          <div className="flex items-center justify-between px-3 py-2 border-b border-white/[0.06] bg-white/[0.02]">
            <span className="text-[11px] font-medium text-[#D1D8E3] flex items-center gap-1.5">
              <Compass className="w-3.5 h-3.5 text-sky-400" />
              {t("tilt.compassViewLabel")}
            </span>
            <div className="flex items-center gap-1.5 font-mono text-[11px]">
              <span className="text-white font-semibold">
                {azDeg.toFixed(0)}° {cardLabel}
              </span>
              <span className="text-[#64748B]">·</span>
              <span className="px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-300 font-semibold border border-sky-400/35">
                FOV H {fovH.toFixed(0)}°
              </span>
            </div>
          </div>

          <div className="relative w-full flex-1 flex items-center justify-center bg-[#090C12]">
            <svg
              ref={compassSvgRef}
              viewBox={`0 0 ${topW} ${topH}`}
              preserveAspectRatio="xMidYMid meet"
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
              className="w-full aspect-[34/25] block select-none touch-none cursor-grab active:cursor-grabbing"
            >
              {/* Concentric physical range rings (FIXED scale so FOV H change is 100% visually true!) */}
              <circle cx={cx} cy={cy} r={maxR} fill="#0D121C" stroke="#334155" strokeWidth="1.2" />
              <circle cx={cx} cy={cy} r={maxR * 0.66} fill="none" stroke="#1E293B" strokeDasharray="3 3" strokeWidth="1" />
              <circle cx={cx} cy={cy} r={maxR * 0.33} fill="none" stroke="#1E293B" strokeDasharray="3 3" strokeWidth="1" />

              {/* Crosshairs & 15° compass ticks */}
              <line x1={cx} y1={cy - maxR} x2={cx} y2={cy + maxR} stroke="#1E293B" strokeWidth="1" />
              <line x1={cx - maxR} y1={cy} x2={cx + maxR} y2={cy} stroke="#1E293B" strokeWidth="1" />
              {Array.from({ length: 24 }).map((_, i) => {
                const a = rad(i * 15);
                const isMajor = i % 3 === 0;
                const rIn = maxR - (isMajor ? 6 : 3);
                return (
                  <line
                    key={i}
                    x1={cx + Math.sin(a) * rIn}
                    y1={cy - Math.cos(a) * rIn}
                    x2={cx + Math.sin(a) * maxR}
                    y2={cy - Math.cos(a) * maxR}
                    stroke={isMajor ? "#475569" : "#1E293B"}
                    strokeWidth={isMajor ? 1.2 : 0.9}
                  />
                );
              })}

              {/* Full TRUE Horizontal FOV Sector (widens/narrows directly with fovH 10°..170°!) */}
              <path d={topGeom.fullSectorPath} fill="#38BDF8" fillOpacity="0.13" stroke="none" />

              {/* Projected Cloud Footprint inside the Horizontal FOV */}
              <path d={topGeom.footprintPath} fill="#38BDF8" fillOpacity="0.28" stroke="#38BDF8" strokeOpacity="0.85" strokeWidth="1.3" />

              {/* Crisp Left & Right Horizontal FOV Boundary Rays */}
              <line x1={cx} y1={cy} x2={topGeom.leftEdge[0]} y2={topGeom.leftEdge[1]} stroke="#38BDF8" strokeWidth="1.8" />
              <line x1={cx} y1={cy} x2={topGeom.rightEdge[0]} y2={topGeom.rightEdge[1]} stroke="#38BDF8" strokeWidth="1.8" />

              {/* Outer FOV H Angular Span Arc */}
              <path d={topGeom.fovSpanArc} fill="none" stroke="#7DD3FC" strokeWidth="2.4" strokeLinecap="round" />

              {/* Center Azimuth Heading Ray & Draggable Handle */}
              <line x1={cx} y1={cy} x2={topGeom.axisEnd[0]} y2={topGeom.axisEnd[1]} stroke="#E0F2FE" strokeOpacity="0.7" strokeWidth="1.3" strokeDasharray="4 3" />
              <circle cx={topGeom.aimPt[0]} cy={topGeom.aimPt[1]} r="3.5" fill="#38BDF8" />
              <circle cx={topGeom.handlePt[0]} cy={topGeom.handlePt[1]} r="7" fill="#38BDF8" fillOpacity="0.22" stroke="#38BDF8" strokeWidth="1.6" />
              <circle cx={topGeom.handlePt[0]} cy={topGeom.handlePt[1]} r="2.6" fill="#FFFFFF" />

              {/* Camera Center Origin */}
              <circle cx={cx} cy={cy} r="4.5" fill="#FFFFFF" stroke="#38BDF8" strokeWidth="2" />

              {/* Sun azimuth dot */}
              {solar && typeof solar.azimuth_deg === "number" && (
                <g transform={`translate(${cx + Math.sin(rad(solar.azimuth_deg)) * (maxR - 10)}, ${cy - Math.cos(rad(solar.azimuth_deg)) * (maxR - 10)})`}>
                  <circle cx="0" cy="0" r="6" fill="#FBBF24" fillOpacity="0.25" />
                  <circle cx="0" cy="0" r="3.4" fill="#FBBF24" stroke="#090C12" strokeWidth="1" />
                </g>
              )}

              {/* Range labels */}
              <text x={cx + 4} y={cy - maxR * 0.66 + 10} fill="#475569" fontSize="8.5" fontFamily="monospace">
                {Math.round(topGeom.maxRangeKm * 0.66)} km
              </text>
              <text x={cx + 4} y={cy - maxR + 11} fill="#64748B" fontSize="8.5" fontFamily="monospace">
                {topGeom.maxRangeKm} km
              </text>

              {/* Cardinal Labels */}
              <text x={cx} y={cy - maxR - 12} textAnchor="middle" fill="#38BDF8" fontSize="11" fontWeight="700" fontFamily="monospace">
                N
              </text>
              <text x={cx + maxR + 15} y={cy + 4} textAnchor="middle" fill="#94A3B8" fontSize="10.5" fontWeight="600" fontFamily="monospace">
                E
              </text>
              <text x={cx} y={cy + maxR + 18} textAnchor="middle" fill="#94A3B8" fontSize="10.5" fontWeight="600" fontFamily="monospace">
                S
              </text>
              <text x={cx - maxR - 15} y={cy + 4} textAnchor="middle" fill="#94A3B8" fontSize="10.5" fontWeight="600" fontFamily="monospace">
                {tCardinal("W")}
              </text>

              {/* Bottom corner footprint width readout */}
              <text x="12" y={topH - 8} fill="#64748B" fontSize="9.5" fontFamily="monospace">
                {isIt ? "Ampiezza" : "Width"} ~{topGeom.widthAtCenterKm.toFixed(1)} km
              </text>
              <text x={topW - 12} y={topH - 8} textAnchor="end" fill="#7DD3FC" fontSize="9.5" fontFamily="monospace">
                ±{(fovH / 2).toFixed(0)}°
              </text>
            </svg>
          </div>
        </div>
      </div>

      {/* Unified 4-Column Telemetry Strip (Clean, Zero Clutter) */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 text-[11px] font-mono tabular-nums">
        <div className="rounded-lg bg-[#0B0E14] border border-white/[0.07] px-3 py-2">
          <div className="text-[10px] text-[#64748B] font-sans uppercase tracking-wider">
            {isIt ? "Centro Ottico" : "Optical Center"}
          </div>
          <div className="text-sky-300 font-semibold mt-0.5">
            {elDeg >= 88.5 ? t("tilt.overhead") : `${sideGeom.centerDistKm.toFixed(1)} km`}
          </div>
        </div>
        <div className="rounded-lg bg-[#0B0E14] border border-white/[0.07] px-3 py-2">
          <div className="text-[10px] text-[#64748B] font-sans uppercase tracking-wider">
            {isIt ? "Area Proiettata" : "Projected Sky"}
          </div>
          <div className="text-[#E2E8F0] font-semibold mt-0.5">
            {sideGeom.nearDistKm.toFixed(1)}–{sideGeom.farDistKm.toFixed(1)} km · {fovH.toFixed(0)}°×{fovV.toFixed(0)}°
          </div>
        </div>
        <div className="rounded-lg bg-[#0B0E14] border border-white/[0.07] px-3 py-2">
          <div className="text-[10px] text-[#64748B] font-sans uppercase tracking-wider flex items-center gap-1">
            <Sun className="w-3 h-3 text-amber-400" /> {isIt ? "Sole" : "Sun"}
          </div>
          <div className="text-amber-300 font-semibold mt-0.5">
            {solar && typeof solar.elevation_deg === "number"
              ? `El ${solar.elevation_deg.toFixed(0)}° · Az ${Number(solar.azimuth_deg ?? 0).toFixed(0)}°${solar.in_camera_fov ? " (FOV)" : ""}`
              : "—"}
          </div>
        </div>
        <div
          className={`rounded-lg border px-3 py-2 ${
            sideGeom.belowHorizonPct > 0
              ? "bg-rose-500/10 border-rose-400/30"
              : "bg-[#0B0E14] border-white/[0.07]"
          }`}
        >
          <div className="text-[10px] text-[#64748B] font-sans uppercase tracking-wider">
            {isIt ? "Orizzonte" : "Horizon"}
          </div>
          <div className={`font-semibold mt-0.5 ${sideGeom.belowHorizonPct > 0 ? "text-rose-300" : "text-emerald-300"}`}>
            {sideGeom.belowHorizonPct > 0
              ? t("tilt.belowHorizonWarn", { pct: sideGeom.belowHorizonPct })
              : isIt
              ? "100% sopra orizzonte"
              : "100% above horizon"}
          </div>
        </div>
      </div>

      {/* 3-Column Precision Control Deck */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        {/* Column 1: Tilt & Roll */}
        <div className="rounded-xl bg-[#0B0E14] border border-white/[0.08] p-3 space-y-2.5">
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[#CBD5E1] font-medium">{t("tilt.tiltFromVertical")}</span>
              <span className="font-mono font-semibold text-sky-300 tabular-nums">
                {tiltDeg.toFixed(0)}° <span className="text-[#64748B] font-normal">(El {elDeg.toFixed(0)}°)</span>
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
            <div className="flex flex-wrap gap-1">
              {[0, 20, 45, 65, 80, 90].map((presetTilt) => {
                const active = Math.abs(tiltDeg - presetTilt) <= 1;
                return (
                  <button
                    key={presetTilt}
                    type="button"
                    onClick={() => updateField({ orientation_elevation_deg: 90 - presetTilt })}
                    className={`px-2 py-0.5 rounded text-[10px] font-mono border transition ${
                      active
                        ? "bg-sky-500/20 border-sky-400/50 text-sky-200"
                        : "bg-[#111520] border-white/[0.08] text-[#9AA1AD] hover:text-white"
                    }`}
                  >
                    {presetTilt}°
                  </button>
                );
              })}
            </div>
          </div>

          <div className="pt-2 border-t border-white/[0.06] space-y-1">
            <div className="flex items-center justify-between gap-2 text-[11px]">
              <span className="text-[#94A3B8]">{t("tilt.rollLabel")}</span>
              <input
                type="number"
                min={-45}
                max={45}
                step={0.5}
                value={rollDeg}
                onChange={(e) => updateField({ orientation_roll_deg: clamp(Number(e.target.value) || 0, -45, 45) })}
                className="obs-input w-16 py-0.5 text-right font-mono tabular-nums text-[11px]"
              />
            </div>
            <input
              type="range"
              min={-25}
              max={25}
              step={0.5}
              value={clamp(rollDeg, -25, 25)}
              onChange={(e) => updateField({ orientation_roll_deg: Number(e.target.value) })}
              className="w-full accent-emerald-400"
            />
          </div>
        </div>

        {/* Column 2: Compass Azimuth */}
        <div className="rounded-xl bg-[#0B0E14] border border-white/[0.08] p-3 space-y-2.5 flex flex-col justify-between">
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[#CBD5E1] font-medium">{t("tilt.azimuthLabel")}</span>
              <span className="font-mono font-semibold text-white tabular-nums">
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
          </div>

          <div className="grid grid-cols-4 gap-1.5">
            {CARDINAL_ANGLES.map((c) => {
              const active = Math.abs(((azDeg - c.deg + 540) % 360) - 180) <= 10;
              return (
                <button
                  key={c.deg}
                  type="button"
                  onClick={() => updateField({ orientation_azimuth_deg: c.deg })}
                  className={`py-1.5 rounded-md text-[11px] font-mono border transition ${
                    active
                      ? "bg-sky-500/20 border-sky-400/50 text-sky-200 font-semibold"
                      : "bg-[#111520] border-white/[0.08] text-[#9AA1AD] hover:text-white"
                  }`}
                >
                  {tCardinal(c.key)}
                </button>
              );
            })}
          </div>
        </div>

        {/* Column 3: Optics — Live Interactive Horizontal & Vertical FOV Sliders + Presets + Cloud Ceiling */}
        <div className="rounded-xl bg-[#0B0E14] border border-white/[0.08] p-3 space-y-2.5">
          {/* Horizontal FOV Slider + Exact Input */}
          <div className="space-y-1">
            <div className="flex items-center justify-between gap-2">
              <span className="text-[#CBD5E1] font-medium flex items-center gap-1">
                <Maximize2 className="w-3 h-3 text-sky-400" />
                {t("tilt.fovH")}
              </span>
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  title="Blocca rapporto 16:9 tra FOV Orizzontale e Verticale"
                  onClick={() => {
                    const next = !lockAspect16x9;
                    setLockAspect16x9(next);
                    if (next) {
                      updateField({ fov_vertical_deg: derive16x9VerticalFov(fovH) });
                    }
                  }}
                  className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono border transition ${
                    lockAspect16x9
                      ? "bg-sky-500/20 border-sky-400/50 text-sky-200"
                      : "bg-[#111520] border-white/[0.08] text-[#64748B] hover:text-[#CBD5E1]"
                  }`}
                >
                  {lockAspect16x9 ? <Link2 className="w-2.5 h-2.5" /> : <Unlink2 className="w-2.5 h-2.5" />}
                  16:9
                </button>
                <input
                  type="number"
                  min={10}
                  max={170}
                  value={Math.round(fovH)}
                  onChange={(e) => setFovH(Number(e.target.value) || 72)}
                  className="obs-input w-14 py-0.5 text-right font-mono tabular-nums text-[11px] text-sky-300"
                />
              </div>
            </div>
            <input
              type="range"
              min={15}
              max={170}
              step={1}
              value={Math.round(fovH)}
              onChange={(e) => setFovH(Number(e.target.value))}
              className="w-full accent-sky-400"
            />
          </div>

          {/* Vertical FOV Slider + Exact Input */}
          <div className="space-y-1">
            <div className="flex items-center justify-between gap-2">
              <span className="text-[#94A3B8] text-[11px]">{t("tilt.fovV")}</span>
              <input
                type="number"
                min={10}
                max={150}
                value={Math.round(fovV)}
                onChange={(e) => {
                  setLockAspect16x9(false);
                  updateField({ fov_vertical_deg: clamp(Number(e.target.value) || 45, 10, 150) });
                }}
                className="obs-input w-14 py-0.5 text-right font-mono tabular-nums text-[11px] text-indigo-300"
              />
            </div>
            <input
              type="range"
              min={10}
              max={140}
              step={1}
              value={Math.round(fovV)}
              onChange={(e) => {
                setLockAspect16x9(false);
                updateField({ fov_vertical_deg: Number(e.target.value) });
              }}
              className="w-full accent-indigo-400"
            />
          </div>

          {/* Lens Quick Presets */}
          <div className="flex flex-wrap gap-1">
            {LENS_PRESETS.map((p) => {
              const active = Math.abs(fovH - p.fovH) <= 2;
              return (
                <button
                  key={p.label}
                  type="button"
                  onClick={() => updateField({ fov_horizontal_deg: p.fovH, fov_vertical_deg: p.fovV })}
                  className={`px-1.5 py-0.5 rounded text-[10px] font-mono border transition ${
                    active
                      ? "bg-sky-500/20 border-sky-400/50 text-sky-200"
                      : "bg-[#111520] border-white/[0.08] text-[#9AA1AD] hover:text-white"
                  }`}
                >
                  {p.label}
                </button>
              );
            })}
          </div>

          {/* Cloud Ceiling Altitude */}
          <div className="pt-2 border-t border-white/[0.06] flex items-center justify-between gap-2" title={t("tilt.cloudAltHint")}>
            <div className="flex items-center gap-1.5">
              <span className="text-[#94A3B8] text-[11px]">{t("tilt.cloudAltM")}</span>
              {value.assumed_cloud_altitude_m !== null ? (
                <button
                  type="button"
                  onClick={() => updateField({ assumed_cloud_altitude_m: null })}
                  className="text-[9.5px] font-mono text-sky-300 hover:underline"
                >
                  {t("tilt.autoLclBtn")}
                </button>
              ) : (
                <span className="text-[9.5px] font-mono px-1.5 py-0.2 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-400/25">
                  AUTO
                </span>
              )}
            </div>
            <input
              type="number"
              min={100}
              max={15000}
              step={100}
              placeholder={`~${Math.round(effectiveCloudAltitudeM || 2000)}`}
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
              className="obs-input w-24 py-0.5 text-right font-mono tabular-nums text-[11px]"
            />
          </div>
        </div>
      </div>

      {/* Live Camera Photo with 3D Horizon, Elevation Iso-Curves, Azimuth Meridians & Sun Overlay */}
      {showLiveCam && (
        <div className="rounded-xl bg-[#0B0E14] border border-white/[0.08] p-3 space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2 text-[11px]">
            <span className="font-medium text-white flex items-center gap-1.5">
              <Eye className="w-3.5 h-3.5 text-sky-400" /> {t("tilt.liveCamPreview")}
            </span>
            <span className="font-mono text-[#8E95A2]">
              Tilt {tiltDeg.toFixed(0)}° (El {elDeg.toFixed(0)}°) · Az {azDeg.toFixed(0)}° {cardLabel} · FOV {fovH.toFixed(0)}°×{fovV.toFixed(0)}°
            </span>
          </div>
          <div className="relative aspect-video max-h-[290px] mx-auto bg-[#08090B] rounded-lg overflow-hidden border border-white/[0.08]">
            <img
              src={camPreviewRetry > 0 ? `/api/camera/latest?layer=raw&r=${camPreviewRetry}` : "/api/camera/latest?layer=raw"}
              alt="Live camera calibration feed"
              className="w-full h-full object-contain transition-opacity duration-200"
              onLoad={(e) => {
                (e.currentTarget as HTMLImageElement).style.opacity = "1";
              }}
              onError={(e) => {
                (e.currentTarget as HTMLImageElement).style.opacity = "0.25";
                if (camPreviewRetry < 3) {
                  window.setTimeout(() => setCamPreviewRetry((n) => n + 1), 2500);
                }
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
