import React, { useState } from "react";
import { Layers, ShieldCheck, Wind } from "lucide-react";
import { ProbabilityBars } from "../components/Charts";
import { ExclusionMaskPainter } from "../components/ExclusionMaskPainter";
import { useI18n } from "../i18n/I18nContext";
import type { Observation } from "../types/api";

export const LiveVisionPage: React.FC<{ obs: Observation | null }> = ({ obs }) => {
  const { t, tEnum } = useI18n();
  const [layer, setLayer] = useState<"raw" | "segmentation" | "motion" | "mask" | "paint_exclude">("segmentation");

  if (!obs) {
    return (
      <div className="bg-[#0F1623] border border-slate-800 rounded-xl p-10 text-center text-sm text-slate-400">
        {t("vision.empty")}
      </div>
    );
  }

  const imageUrl =
    layer === "mask"
      ? `/api/camera/latest?layer=mask&t=${encodeURIComponent(obs.captured_at)}`
      : `/api/analysis/image/${obs.id}?layer=${layer}&t=${encodeURIComponent(obs.captured_at)}`;

  const classProbs = Object.entries(obs.classification_probabilities || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([label, value]) => ({ label: tEnum(label), value }));

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-white">{t("vision.title")}</h2>
          <p className="text-xs text-slate-400">
            {t("vision.subtitle", { id: obs.id, time: new Date(obs.captured_at).toLocaleString() })}
          </p>
        </div>
        <div className="inline-flex flex-wrap bg-[#0F1623] border border-slate-800 rounded-lg p-1 gap-1">
          {(
            [
              ["raw", t("vision.layer.raw")],
              ["segmentation", t("vision.layer.segmentation")],
              ["motion", t("vision.layer.motion")],
              ["mask", t("vision.layer.mask")],
              ["paint_exclude", t("vision.layer.paintExclude")],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              onClick={() => setLayer(id)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition ${
                layer === id ? "bg-sky-600 text-white" : "text-slate-300 hover:bg-slate-800"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
        <div className="xl:col-span-2 bg-[#0F1623] border border-slate-800 rounded-xl overflow-hidden">
          {layer === "paint_exclude" ? (
            <div className="p-4">
              <ExclusionMaskPainter
                imageUrl={`/api/analysis/image/${obs.id}?layer=raw&t=${encodeURIComponent(obs.captured_at)}`}
              />
            </div>
          ) : (
            <div className="aspect-video bg-[#060911] flex items-center justify-center">
              <img src={imageUrl} alt={`Vision layer ${layer}`} className="w-full h-full object-contain" />
            </div>
          )}
          {layer === "segmentation" && (
            <div className="px-4 py-2.5 border-t border-slate-800 flex flex-wrap gap-4 text-xs text-slate-300">
              <span className="inline-flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-[#238CDC]" /> {t("vision.legend.clear")} (
                {obs.clear_sky_pct.toFixed(1)}%)
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-[#BEEBFA]" /> {t("vision.legend.thin")} (
                {obs.thin_cloud_pct.toFixed(1)}%)
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-[#F5F5F5]" /> {t("vision.legend.dense")} (
                {obs.dense_cloud_pct.toFixed(1)}%)
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-[#69555F]" /> {t("vision.legend.dark")} (
                {obs.dark_cloud_pct.toFixed(1)}%)
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-[#D2C8A0]" /> {t("vision.legend.haze")} (
                {obs.haze_fog_pct.toFixed(1)}%)
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-[#FFAF3C]" /> {t("vision.legend.uncertain")} (
                {obs.uncertainty_pct.toFixed(1)}%)
              </span>
            </div>
          )}
        </div>

        <div className="space-y-4">
          <div className="bg-[#0F1623] border border-slate-800 rounded-xl p-4 space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-white flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-400" /> {t("vision.imageQuality")}
              </span>
              <span
                className={`px-2 py-0.5 rounded text-[11px] font-mono ${
                  obs.is_usable ? "bg-emerald-500/20 text-emerald-300" : "bg-rose-500/20 text-rose-300"
                }`}
              >
                {(obs.quality_score * 100).toFixed(0)}% · {obs.is_usable ? t("vision.usable") : t("vision.flagged")}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-slate-900/70 rounded p-2">
                <span className="text-slate-400">{t("vision.luminance")}</span>
                <div className="font-mono text-white">{obs.quality?.mean_luminance ?? obs.avg_luminance}</div>
              </div>
              <div className="bg-slate-900/70 rounded p-2">
                <span className="text-slate-400">{t("vision.sharpness")}</span>
                <div className="font-mono text-white">{obs.quality?.laplacian_var_full ?? "—"}</div>
              </div>
              <div className="bg-slate-900/70 rounded p-2">
                <span className="text-slate-400">{t("vision.cameraShift")}</span>
                <div className="font-mono text-white">{obs.quality?.camera_shift_px ?? 0} px</div>
              </div>
              <div className="bg-slate-900/70 rounded p-2">
                <span className="text-slate-400">{t("vision.glareRatio")}</span>
                <div className="font-mono text-white">{((obs.quality?.glare_ratio ?? 0) * 100).toFixed(1)}%</div>
              </div>
            </div>
          </div>

          <div className="bg-[#0F1623] border border-slate-800 rounded-xl p-4 space-y-2.5">
            <div className="flex items-center gap-2 text-sm font-semibold text-white">
              <Layers className="w-4 h-4 text-sky-400" /> {t("vision.regionalGrid")}
            </div>
            <div className="grid grid-cols-3 gap-1.5">
              {(obs.coverage?.grid || []).map((cell, idx) => (
                <div key={idx} className="bg-slate-900 border border-slate-800 rounded p-2 text-center">
                  <div className="text-[10px] text-slate-500">
                    R{cell.row + 1}C{cell.col + 1}
                  </div>
                  <div className="text-xs font-mono font-semibold text-sky-300 mt-0.5">
                    {cell.cloud_pct != null ? `${cell.cloud_pct.toFixed(0)}%` : t("vision.masked")}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-[#0F1623] border border-slate-800 rounded-xl p-4 space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-white flex items-center gap-2">
                <Wind className="w-4 h-4 text-cyan-400" /> {t("vision.cloudProbs")}
              </span>
            </div>
            <ProbabilityBars items={classProbs} />
          </div>
        </div>
      </div>
    </div>
  );
};
