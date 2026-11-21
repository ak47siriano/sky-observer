import React, { useMemo } from "react";

export interface SkyHorizonOverlayProps {
  elevationDeg: number;
  azimuthDeg: number;
  rollDeg?: number;
  fovHorizontalDeg?: number;
  fovVerticalDeg?: number;
  solar?: {
    elevation_deg?: number;
    azimuth_deg?: number;
    is_daylight?: boolean;
  } | null;
  className?: string;
}

const CARDINALS: { label: string; az: number }[] = [
  { label: "N", az: 0 },
  { label: "NE", az: 45 },
  { label: "E", az: 90 },
  { label: "SE", az: 135 },
  { label: "S", az: 180 },
  { label: "SW", az: 225 },
  { label: "W", az: 270 },
  { label: "NW", az: 315 },
];

export const SkyHorizonOverlay: React.FC<SkyHorizonOverlayProps> = ({
  elevationDeg,
  azimuthDeg,
  rollDeg = 0,
  fovHorizontalDeg = 72,
  fovVerticalDeg = 45,
  solar,
  className = "",
}) => {
  const overlay = useMemo(() => {
    const rad = (d: number) => (d * Math.PI) / 180;
    const el = Math.max(-20, Math.min(90, Number(elevationDeg ?? 65)));
    const az = ((Number(azimuthDeg || 0) % 360) + 360) % 360;
    const roll = Math.max(-45, Math.min(45, Number(rollDeg || 0)));
    const fovH = Math.max(10, Math.min(170, Number(fovHorizontalDeg || 72)));
    const fovV = Math.max(10, Math.min(170, Number(fovVerticalDeg || 45)));

    const alpha = rad(el);
    const psi = rad(az);
    const rho = rad(roll);

    // ENU camera basis
    const fCam = [Math.cos(alpha) * Math.sin(psi), Math.cos(alpha) * Math.cos(psi), Math.sin(alpha)];
    const r0 = [Math.cos(psi), -Math.sin(psi), 0];
    const u0 = [-Math.sin(alpha) * Math.sin(psi), -Math.sin(alpha) * Math.cos(psi), Math.cos(alpha)];
    const cosR = Math.cos(rho);
    const sinR = Math.sin(rho);
    const rCam = [
      cosR * r0[0] - sinR * u0[0],
      cosR * r0[1] - sinR * u0[1],
      cosR * r0[2] - sinR * u0[2],
    ];
    const uCam = [
      sinR * r0[0] + cosR * u0[0],
      sinR * r0[1] + cosR * u0[1],
      sinR * r0[2] + cosR * u0[2],
    ];

    const W = 1000;
    const H = 562.5;
    const fx = W / (2 * Math.tan(rad(fovH) / 2));
    const fy = H / (2 * Math.tan(rad(fovV) / 2));
    const cx = W / 2;
    const cy = H / 2;

    const projectSky = (azDeg: number, elDeg: number): [number, number] | null => {
      const aR = rad(azDeg);
      const eR = rad(elDeg);
      const ray = [Math.cos(eR) * Math.sin(aR), Math.cos(eR) * Math.cos(aR), Math.sin(eR)];
      const cz = ray[0] * fCam[0] + ray[1] * fCam[1] + ray[2] * fCam[2];
      if (cz <= 0.04) return null;
      const cxP = (ray[0] * rCam[0] + ray[1] * rCam[1] + ray[2] * rCam[2]) / cz;
      const cyP = (ray[0] * uCam[0] + ray[1] * uCam[1] + ray[2] * uCam[2]) / cz;
      const x = cx + fx * cxP;
      const y = cy - fy * cyP;
      if (x < -120 || x > W + 120 || y < -120 || y > H + 120) return null;
      return [x, y];
    };

    const sampleElevationCurve = (targetEl: number) => {
      const pts: [number, number][] = [];
      const span = Math.min(180, fovH * 1.15);
      const steps = 36;
      for (let i = 0; i <= steps; i++) {
        const a = az - span / 2 + (span * i) / steps;
        const p = projectSky(a, targetEl);
        if (p) pts.push(p);
      }
      if (pts.length < 2) return null;
      const path = pts.map((p, idx) => `${idx === 0 ? "M" : "L"} ${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
      const mid = pts[Math.floor(pts.length / 2)];
      return { el: targetEl, path, mid, pts };
    };

    const horizonCurve = sampleElevationCurve(0);
    const elCurves = [15, 30, 45, 60, 75]
      .map((e) => sampleElevationCurve(e))
      .filter((c): c is NonNullable<typeof c> => c !== null);

    // Below-horizon shaded polygon if horizon is visible inside frame
    let belowHorizonPoly: string | null = null;
    if (horizonCurve && horizonCurve.pts.length >= 2) {
      const first = horizonCurve.pts[0];
      const last = horizonCurve.pts[horizonCurve.pts.length - 1];
      const polyPts = [
        ...horizonCurve.pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`),
        `${last[0].toFixed(1)},${H + 40}`,
        `${first[0].toFixed(1)},${H + 40}`,
      ];
      belowHorizonPoly = polyPts.join(" ");
    }

    // Compass bearing markers at mid-frame or near horizon
    const refEl = Math.max(4, Math.min(60, el - fovV * 0.28));
    const compassTicks = CARDINALS.map((c) => {
      const pt = projectSky(c.az, refEl);
      if (!pt || pt[0] < 24 || pt[0] > W - 24 || pt[1] < 20 || pt[1] > H - 16) return null;
      return { ...c, x: pt[0], y: pt[1] };
    }).filter((x): x is NonNullable<typeof x> => x !== null);

    // Zenith marker
    const zenithPt = projectSky(az, 90);

    // Solar marker
    let sunPt: { x: number; y: number; el: number; az: number } | null = null;
    if (solar && typeof solar.elevation_deg === "number" && typeof solar.azimuth_deg === "number") {
      const p = projectSky(solar.azimuth_deg, solar.elevation_deg);
      if (p && p[0] >= -20 && p[0] <= W + 20 && p[1] >= -20 && p[1] <= H + 20) {
        sunPt = { x: p[0], y: p[1], el: solar.elevation_deg, az: solar.azimuth_deg };
      }
    }

    return { W, H, horizonCurve, belowHorizonPoly, elCurves, compassTicks, zenithPt, sunPt };
  }, [azimuthDeg, elevationDeg, fovHorizontalDeg, fovVerticalDeg, rollDeg, solar]);

  return (
    <svg
      viewBox={`0 0 ${overlay.W} ${overlay.H}`}
      preserveAspectRatio="none"
      className={`absolute inset-0 w-full h-full pointer-events-none select-none ${className}`}
    >
      {/* Below-horizon auto-mask warning shade */}
      {overlay.belowHorizonPoly && (
        <polygon points={overlay.belowHorizonPoly} fill="rgba(244, 63, 94, 0.20)" />
      )}

      {/* Elevation angle curves (15°, 30°, 45°, 60°, 75°) */}
      {overlay.elCurves.map((c) => (
        <g key={c.el}>
          <path
            d={c.path}
            fill="none"
            stroke="#38BDF8"
            strokeOpacity="0.48"
            strokeWidth="1.5"
            strokeDasharray="6 5"
          />
          {c.mid[1] > 18 && c.mid[1] < overlay.H - 18 && (
            <text
              x={Math.min(overlay.W - 44, Math.max(24, c.mid[0]))}
              y={c.mid[1] - 5}
              fill="#7DD3FC"
              fontSize="13"
              fontFamily="monospace"
              fontWeight="600"
            >
              {c.el}° El
            </text>
          )}
        </g>
      ))}

      {/* Mathematical 0° Horizon Line */}
      {overlay.horizonCurve && (
        <g>
          <path
            d={overlay.horizonCurve.path}
            fill="none"
            stroke="#FB7185"
            strokeWidth="2.4"
            strokeDasharray="10 5"
          />
          <text
            x={Math.min(overlay.W - 110, Math.max(18, overlay.horizonCurve.mid[0] - 40))}
            y={Math.min(overlay.H - 8, Math.max(18, overlay.horizonCurve.mid[1] - 7))}
            fill="#FDA4AF"
            fontSize="14"
            fontFamily="monospace"
            fontWeight="700"
          >
            HORIZON 0°
          </text>
        </g>
      )}

      {/* Optical Axis Center Reticle */}
      <g transform={`translate(${overlay.W / 2}, ${overlay.H / 2})`} opacity="0.75">
        <circle cx="0" cy="0" r="14" fill="none" stroke="#38BDF8" strokeWidth="1.5" />
        <line x1="-22" y1="0" x2="-8" y2="0" stroke="#38BDF8" strokeWidth="1.5" />
        <line x1="8" y1="0" x2="22" y2="0" stroke="#38BDF8" strokeWidth="1.5" />
        <line x1="0" y1="-22" x2="0" y2="-8" stroke="#38BDF8" strokeWidth="1.5" />
        <line x1="0" y1="8" x2="0" y2="22" stroke="#38BDF8" strokeWidth="1.5" />
      </g>

      {/* Zenith 90° Marker */}
      {overlay.zenithPt && (
        <g transform={`translate(${overlay.zenithPt[0]}, ${overlay.zenithPt[1]})`}>
          <circle cx="0" cy="0" r="7" fill="none" stroke="#38BDF8" strokeWidth="2" />
          <line x1="-11" y1="0" x2="11" y2="0" stroke="#38BDF8" strokeWidth="1.5" />
          <line x1="0" y1="-11" x2="0" y2="11" stroke="#38BDF8" strokeWidth="1.5" />
          <text x="12" y="4" fill="#38BDF8" fontSize="13" fontFamily="monospace" fontWeight="700">
            ZENITH 90°
          </text>
        </g>
      )}

      {/* Compass Cardinals */}
      {overlay.compassTicks.map((t) => (
        <g key={t.label} transform={`translate(${t.x}, ${t.y})`}>
          <rect
            x="-18"
            y="-12"
            width="36"
            height="20"
            rx="4"
            fill="rgba(11, 12, 14, 0.78)"
            stroke="rgba(56, 189, 248, 0.45)"
            strokeWidth="1.2"
          />
          <text x="0" y="2" textAnchor="middle" fill="#F4F5F7" fontSize="13" fontFamily="monospace" fontWeight="700">
            {t.label}
          </text>
        </g>
      ))}

      {/* Astronomical Sun Position & Glare Guard Ring */}
      {overlay.sunPt && (
        <g transform={`translate(${overlay.sunPt.x}, ${overlay.sunPt.y})`}>
          <circle
            cx="0"
            cy="0"
            r="34"
            fill="rgba(251, 191, 36, 0.12)"
            stroke="#FBBF24"
            strokeWidth="1.4"
            strokeDasharray="5 3"
          />
          <circle cx="0" cy="0" r="9" fill="#FBBF24" stroke="#0B0C0E" strokeWidth="2" />
          <text x="14" y="4" fill="#FDE68A" fontSize="13" fontFamily="monospace" fontWeight="700">
            ☀ {overlay.sunPt.el.toFixed(0)}°
          </text>
        </g>
      )}
    </svg>
  );
};
