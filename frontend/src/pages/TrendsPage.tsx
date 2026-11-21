import React, { useEffect, useState } from "react";
import { Download } from "lucide-react";
import { TimeSeriesChart } from "../components/Charts";
import { useI18n } from "../i18n/I18nContext";
import { api } from "../services/api";
import type { Observation } from "../types/api";

export const TrendsPage: React.FC = () => {
  const { t } = useI18n();
  const [hours, setHours] = useState<number>(6);
  const [items, setItems] = useState<Observation[]>([]);
  const [maeTrend, setMaeTrend] = useState<any[]>([]);

  useEffect(() => {
    api.getAnalysisHistory(240, hours).then((d) => setItems(d.items || [])).catch(() => {});
    api.getForecastVerification().then((v) => setMaeTrend(v.mae_trend || [])).catch(() => {});
  }, [hours]);

  const labels = items.map((o) =>
    new Date(o.captured_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
  );
  const maeLabels = maeTrend.map((m) =>
    new Date(m.verified_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-white">{t("trends.title")}</h2>
          <p className="text-xs text-[#8E95A2]">{t("trends.showing", { n: items.length })}</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <a
            href={api.exportAnalyticsUrl("csv", hours)}
            download
            className="obs-btn-secondary inline-flex items-center gap-1.5 px-3 py-1.5 text-xs"
          >
            <Download className="w-3.5 h-3.5 text-emerald-400" /> {t("trends.exportCsv")}
          </a>
          <a
            href={api.exportAnalyticsUrl("json", hours)}
            download
            className="obs-btn-secondary inline-flex items-center gap-1.5 px-3 py-1.5 text-xs"
          >
            <Download className="w-3.5 h-3.5 text-sky-400" /> {t("trends.exportJson")}
          </a>
          <div className="inline-flex bg-[#13151A] border border-white/[0.07] rounded-xl p-1 gap-1">
            {[1, 6, 24, 72].map((h) => (
              <button
                key={h}
                type="button"
                onClick={() => setHours(h)}
                className={`px-3 py-1 rounded-lg text-xs font-mono transition ${
                  hours === h ? "bg-[#F4F5F7] text-[#0B0C0E] font-semibold" : "text-[#8E95A2] hover:text-white"
                }`}
              >
                {h}h
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <TimeSeriesChart
          title={t("trends.chart.cloudCover")}
          labels={labels}
          unit="%"
          series={[
            { label: t("trends.series.totalCloud"), color: "#60A5FA", values: items.map((o) => o.cloud_coverage_pct) },
            { label: t("trends.series.denseCloud"), color: "#E5E7EB", values: items.map((o) => o.dense_cloud_pct) },
            { label: t("trends.series.darkCloud"), color: "#F43F5E", values: items.map((o) => o.dark_cloud_pct) },
          ]}
        />

        <TimeSeriesChart
          title={t("trends.chart.luminance")}
          labels={labels}
          yMin={-20}
          yMax={100}
          series={[
            { label: t("trends.series.lum"), color: "#FBBF24", values: items.map((o) => o.avg_luminance) },
            { label: t("trends.series.bsi"), color: "#3B82F6", values: items.map((o) => o.blue_sky_index * 100) },
          ]}
        />

        <TimeSeriesChart
          title={t("trends.chart.motion")}
          labels={labels}
          yMin={0}
          yMax={100}
          series={[
            { label: t("trends.series.speed"), color: "#22D3EE", values: items.map((o) => o.motion_speed_px_s) },
            { label: t("trends.series.consistency"), color: "#34D399", values: items.map((o) => o.motion_consistency * 100) },
          ]}
        />

        <TimeSeriesChart
          title={t("trends.chart.quality")}
          labels={labels}
          yMin={0}
          yMax={100}
          series={[
            { label: t("trends.series.clsConf"), color: "#A78BFA", values: items.map((o) => o.classification_confidence * 100) },
            { label: t("trends.series.qualScore"), color: "#10B981", values: items.map((o) => o.quality_score * 100) },
          ]}
        />
      </div>

      {maeTrend.length > 0 ? (
        <TimeSeriesChart
          title={t("trends.chart.mae")}
          labels={maeLabels}
          unit="%"
          yMin={0}
          yMax={100}
          series={[
            { label: t("trends.series.absErr"), color: "#F59E0B", values: maeTrend.map((m) => m.abs_error) },
            { label: t("trends.series.rollingMae"), color: "#10B981", values: maeTrend.map((m) => m.rolling_mae_pct) },
          ]}
        />
      ) : (
        <div className="obs-panel p-4 text-xs text-[#8E95A2]">{t("trends.maeEmpty")}</div>
      )}
    </div>
  );
};
