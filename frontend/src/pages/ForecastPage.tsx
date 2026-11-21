import React, { useEffect, useState } from "react";
import { AlertCircle, CheckCircle2, CloudRain, Compass } from "lucide-react";
import { ProbabilityBars } from "../components/Charts";
import { useI18n } from "../i18n/I18nContext";
import { api } from "../services/api";
import type { ForecastItem } from "../types/api";

export const ForecastPage: React.FC<{ forecasts: ForecastItem[] }> = ({ forecasts }) => {
  const { t, tEnum } = useI18n();
  const [verification, setVerification] = useState<any>(null);
  const [history, setHistory] = useState<ForecastItem[]>([]);

  useEffect(() => {
    api.getForecastVerification().then(setVerification).catch(() => {});
    api.getForecastHistory(40).then((d) => setHistory(d.items || [])).catch(() => {});
  }, [forecasts]);

  return (
    <div className="space-y-4">
      <div className="obs-inset px-4 py-3 text-xs text-[#9AA1AD] flex items-center gap-2.5">
        <AlertCircle className="w-4 h-4 text-amber-400 shrink-0" />
        <span>
          <strong className="text-[#F4F5F7]">{t("forecast.disclaimerTitle")}</strong> {t("forecast.disclaimerBody")}
        </span>
      </div>

      {forecasts.length === 0 || forecasts[0]?.status !== "available" ? (
        <div className="obs-panel p-10 text-center text-sm text-[#8E95A2]">{t("forecast.unavailable")}</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
          {forecasts.map((fc) => (
            <div key={fc.id} className="obs-panel p-4 space-y-3">
              <div className="flex items-center justify-between">
                <span className="px-2.5 py-0.5 rounded-md bg-white/[0.08] text-[#F4F5F7] text-xs font-mono font-semibold tabular-nums">
                  +{fc.horizon_minutes} min
                </span>
                <span className="text-xs font-mono tabular-nums text-[#8E95A2]">
                  {t("forecast.conf")} {(fc.confidence * 100).toFixed(0)}%
                </span>
              </div>

              <div>
                <div className="text-base font-semibold text-white">{tEnum(fc.forecast)}</div>
                <div className="text-xs text-[#8E95A2] mt-0.5">
                  {t("forecast.predCover")}{" "}
                  <span className="font-mono tabular-nums text-sky-300 font-semibold">
                    {fc.cloud_cover_prediction.toFixed(1)}%
                  </span>
                </div>
              </div>

              <ProbabilityBars
                items={[
                  { label: t("forecast.clearing"), value: fc.clearing_prob, color: "#34D399" },
                  { label: t("forecast.increasing"), value: fc.increasing_prob, color: "#60A5FA" },
                  { label: t("forecast.overcast"), value: fc.overcast_prob, color: "#94A3B8" },
                  { label: t("forecast.darkening"), value: fc.darkening_prob, color: "#F59E0B" },
                  { label: t("forecast.precipExp"), value: fc.precip_signal_prob, color: "#818CF8" },
                ]}
              />

              <div className="border-t border-white/[0.06] pt-2.5 space-y-1.5">
                <div className="obs-label">{t("forecast.signals")}</div>
                <div className="flex flex-wrap gap-1">
                  {fc.signals.map((sig) => (
                    <span key={sig} className="px-1.5 py-0.5 rounded bg-white/[0.06] text-[10px] font-mono text-[#C5CBD5]">
                      {tEnum(sig)}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="obs-panel p-5 space-y-3">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <Compass className="w-4 h-4 text-sky-400" /> {t("forecast.evidenceTitle")}
          </h3>
          {forecasts[0]?.reasons?.length ? (
            <ul className="space-y-1.5 text-xs text-[#C5CBD5] list-disc list-inside">
              {forecasts[0].reasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-[#646B78]">{t("forecast.noReasoning")}</p>
          )}
          <div className="pt-2 border-t border-white/[0.06] text-[11px] text-[#8E95A2]">
            {t("forecast.modelVersion")}{" "}
            <span className="font-mono text-[#D2D6DF]">{forecasts[0]?.model_version || "—"}</span>
          </div>
        </div>

        <div className="obs-panel p-5 space-y-3">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" /> {t("forecast.verificationTitle")}
          </h3>
          {!verification || verification.total_verified === 0 ? (
            <p className="text-xs text-[#8E95A2]">{t("forecast.verificationEmpty")}</p>
          ) : (
            <div className="space-y-3">
              <div className="grid grid-cols-3 sm:grid-cols-5 gap-2 text-xs">
                <div className="obs-inset p-2.5">
                  <div className="obs-label">{t("forecast.verified")}</div>
                  <div className="font-mono tabular-nums font-bold text-white mt-0.5">{verification.total_verified}</div>
                </div>
                <div className="obs-inset p-2.5">
                  <div className="obs-label">{t("forecast.mae")}</div>
                  <div className="font-mono tabular-nums font-bold text-sky-300 mt-0.5">{verification.overall.mae_pct}%</div>
                </div>
                <div className="obs-inset p-2.5">
                  <div className="obs-label">{t("forecast.rmse")}</div>
                  <div className="font-mono tabular-nums font-bold text-sky-300 mt-0.5">{verification.overall.rmse_pct}%</div>
                </div>
                <div className="obs-inset p-2.5">
                  <div className="obs-label">{t("forecast.accuracy")}</div>
                  <div className="font-mono tabular-nums font-bold text-emerald-300 mt-0.5">
                    {(verification.overall.accuracy * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="obs-inset p-2.5">
                  <div className="obs-label">{t("forecast.brier")}</div>
                  <div className="font-mono tabular-nums font-bold text-amber-300 mt-0.5">{verification.overall.brier_score}</div>
                </div>
              </div>
              {verification.calibration_bins?.length > 0 && (
                <div>
                  <div className="obs-label mb-1.5">{t("forecast.calibration")}</div>
                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-[11px] font-mono tabular-nums">
                    {verification.calibration_bins.map((b: any) => (
                      <div key={b.bin} className="obs-inset p-2">
                        <div className="text-[#8E95A2]">{b.bin}</div>
                        <div className="text-emerald-300">Acc {(b.empirical_accuracy * 100).toFixed(0)}%</div>
                        <div className="text-[#646B78]">n={b.count}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="obs-panel p-5">
        <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
          <CloudRain className="w-4 h-4 text-sky-400" /> {t("forecast.historyTitle")}
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="text-[#8E95A2] border-b border-white/[0.08]">
              <tr>
                <th className="py-2 pr-3">{t("forecast.col.generated")}</th>
                <th className="py-2 pr-3">{t("forecast.col.horizon")}</th>
                <th className="py-2 pr-3">{t("forecast.col.condition")}</th>
                <th className="py-2 pr-3">{t("forecast.col.predCover")}</th>
                <th className="py-2 pr-3">{t("forecast.col.conf")}</th>
                <th className="py-2 pr-3">{t("forecast.col.verified")}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.05]">
              {history.slice(0, 20).map((f) => (
                <tr key={f.id}>
                  <td className="py-2 pr-3 font-mono tabular-nums text-[#C5CBD5]">
                    {new Date(f.generated_at).toLocaleString()}
                  </td>
                  <td className="py-2 pr-3 font-mono tabular-nums text-sky-300">+{f.horizon_minutes}m</td>
                  <td className="py-2 pr-3 text-[#F4F5F7]">{tEnum(f.forecast)}</td>
                  <td className="py-2 pr-3 font-mono tabular-nums">{f.cloud_cover_prediction.toFixed(1)}%</td>
                  <td className="py-2 pr-3 font-mono tabular-nums">{(f.confidence * 100).toFixed(0)}%</td>
                  <td className="py-2 pr-3">{f.verified ? t("forecast.yes") : t("forecast.pending")}</td>
                </tr>
              ))}
              {history.length === 0 && (
                <tr>
                  <td colSpan={6} className="py-6 text-center text-[#646B78]">
                    {t("forecast.noHistory")}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
