import React from "react";
import { useI18n } from "../i18n/I18nContext";

export interface SeriesConfig {
  label: string;
  color: string;
  values: number[];
}

export const TimeSeriesChart: React.FC<{
  title: string;
  labels: string[];
  series: SeriesConfig[];
  yMin?: number;
  yMax?: number;
  unit?: string;
  height?: number;
}> = ({ title, labels, series, yMin = 0, yMax = 100, unit = "", height = 190 }) => {
  const { t } = useI18n();
  const width = 620;
  const padL = 42;
  const padR = 14;
  const padT = 16;
  const padB = 26;
  const innerW = width - padL - padR;
  const innerH = height - padT - padB;
  const span = Math.max(yMax - yMin, 1e-4);

  const count = labels.length;
  const xFor = (i: number) => padL + (count <= 1 ? innerW / 2 : (i / (count - 1)) * innerW);
  const yFor = (v: number) => padT + innerH - ((Math.max(yMin, Math.min(yMax, v)) - yMin) / span) * innerH;
  const baseLineY = padT + innerH;

  return (
    <div className="obs-panel p-5">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3.5">
        <h3 className="text-sm font-medium text-[#F4F5F7]">{title}</h3>
        <div className="flex flex-wrap items-center gap-3 text-xs">
          {series.map((s) => (
            <span key={s.label} className="inline-flex items-center gap-1.5 text-[#9AA1AD]">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: s.color }} />
              {s.label}
              {s.values.length > 0 && (
                <span className="font-mono tabular-nums text-[#F4F5F7]">
                  {s.values[s.values.length - 1].toFixed(1)}
                  {unit}
                </span>
              )}
            </span>
          ))}
        </div>
      </div>

      {count === 0 ? (
        <div className="h-[140px] flex items-center justify-center text-xs text-[#646B78]">
          {t("chart.waiting")}
        </div>
      ) : (
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto overflow-visible">
          <defs>
            {series.map((s, idx) => (
              <linearGradient key={s.label} id={`grad-${idx}-${height}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={s.color} stopOpacity="0.16" />
                <stop offset="100%" stopColor={s.color} stopOpacity="0" />
              </linearGradient>
            ))}
          </defs>

          {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
            const y = padT + innerH * (1 - tick);
            const val = yMin + span * tick;
            return (
              <g key={tick}>
                <line x1={padL} y1={y} x2={width - padR} y2={y} stroke="rgba(255,255,255,0.06)" strokeDasharray="2 3" />
                <text x={padL - 6} y={y + 3} textAnchor="end" fontSize="10" fill="#646B78" fontFamily="monospace">
                  {val.toFixed(0)}
                </text>
              </g>
            );
          })}

          {series.map((s, idx) => {
            if (s.values.length === 0) return null;
            const coords = s.values.map((v, i) => ({ x: xFor(i), y: yFor(v) }));
            const pts = coords.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
            const areaPts =
              coords.length >= 2
                ? `${coords[0].x.toFixed(1)},${baseLineY.toFixed(1)} ${pts} ${coords[coords.length - 1].x.toFixed(1)},${baseLineY.toFixed(1)}`
                : "";
            return (
              <g key={s.label}>
                {areaPts && <polygon fill={`url(#grad-${idx}-${height})`} points={areaPts} />}
                <polyline fill="none" stroke={s.color} strokeWidth="1.9" strokeLinejoin="round" points={pts} />
                {s.values.length <= 40 &&
                  s.values.map((v, i) => <circle key={i} cx={xFor(i)} cy={yFor(v)} r="2.1" fill={s.color} />)}
              </g>
            );
          })}

          {count >= 1 && (
            <>
              <text x={padL} y={height - 6} fontSize="10" fill="#646B78" fontFamily="monospace">
                {labels[0]}
              </text>
              {count >= 3 && (
                <text x={padL + innerW / 2} y={height - 6} textAnchor="middle" fontSize="10" fill="#646B78" fontFamily="monospace">
                  {labels[Math.floor(count / 2)]}
                </text>
              )}
              <text x={width - padR} y={height - 6} textAnchor="end" fontSize="10" fill="#646B78" fontFamily="monospace">
                {labels[count - 1]}
              </text>
            </>
          )}
        </svg>
      )}
    </div>
  );
};

export const ProbabilityBars: React.FC<{ items: Array<{ label: string; value: number; color?: string }> }> = ({ items }) => (
  <div className="space-y-2.5">
    {items.map((it) => {
      const pct = Math.max(0, Math.min(100, it.value * 100));
      return (
        <div key={it.label} className="space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-[#9AA1AD]">{it.label}</span>
            <span className="font-mono tabular-nums text-[#F4F5F7]">{pct.toFixed(0)}%</span>
          </div>
          <div className="h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{ width: `${pct}%`, backgroundColor: it.color || "#60A5FA" }}
            />
          </div>
        </div>
      );
    })}
  </div>
);
