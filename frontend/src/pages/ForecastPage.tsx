import React, { useEffect, useState } from "react";
import { AlertCircle, AlertTriangle, CheckCircle2, CloudRain, Compass, Layers } from "lucide-react";
import { ProbabilityBars } from "../components/Charts";
import { useI18n } from "../i18n/I18nContext";
import { api } from "../services/api";
import type { ForecastItem } from "../types/api";

export const ForecastPage: React.FC<{ forecasts: ForecastItem[] }> = ({ forecasts }) => {
  const { t, tEnum, locale } = useI18n();
  const isIt = locale === "it";
  const [verification, setVerification] = useState<any>(null);
  const [history, setHistory] = useState<ForecastItem[]>([]);
  const [advHorizon, setAdvHorizon] = useState<number>(15);
  const [advNonce, setAdvNonce] = useState<number>(() => Date.now());
  const [advErr, setAdvErr] = useState(false);

  useEffect(() => {
    api.getForecastVerification().then(setVerification).catch(() => {});
    api.getForecastHistory(40).then((d) => setHistory(d.items || [])).catch(() => {});
    setAdvNonce(Date.now());
    setAdvErr(false);
  }, [forecasts]);

  const formatSkill = (s: number | null | undefined) => {
    if (s == null || Number.isNaN(Number(s))) return isIt ? "N/D" : "N/A";
    const pct = Number(s) * 100;
    return `${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%`;
  };

  const selectedAdvFc =
    forecasts.find((f) => f.horizon_minutes === advHorizon) || forecasts[0] || null;

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
          {forecasts.map((fc) => {
            const p10 = fc.cover_p10 ?? fc.baselines?.cover_p10;
            const p90 = fc.cover_p90 ?? fc.baselines?.cover_p90;
            const fovExit = fc.baselines?.fov_exit_fraction;
            const isLowRel = Boolean(fc.low_reliability || fc.baselines?.low_reliability);
            return (
              <div key={fc.id} className="obs-panel p-4 space-y-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="px-2.5 py-0.5 rounded-md bg-white/[0.08] text-[#F4F5F7] text-xs font-mono font-semibold tabular-nums">
                    +{fc.horizon_minutes} min
                  </span>
                  <div className="flex items-center gap-1.5">
                    {isLowRel && (
                      <span
                        title={fc.baselines?.low_reliability_reason || t("forecast.lowReliability")}
                        className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-amber-500/15 border border-amber-500/30 text-amber-300 text-[10px] font-mono"
                      >
                        <AlertTriangle className="w-3 h-3" />
                        {t("forecast.lowReliability")}
                      </span>
                    )}
                    <span className="text-xs font-mono tabular-nums text-[#8E95A2]">
                      {t("forecast.conf")} {(fc.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                <div>
                  <div className="text-base font-semibold text-white">{tEnum(fc.forecast)}</div>
                  <div className="text-xs text-[#8E95A2] mt-0.5 flex flex-wrap items-baseline gap-x-2">
                    <span>
                      {t("forecast.predCover")}{" "}
                      <span className="font-mono tabular-nums text-sky-300 font-semibold">
                        {fc.cloud_cover_prediction.toFixed(1)}%
                      </span>
                    </span>
                    {p10 != null && p90 != null && (
                      <span className="text-[11px] font-mono tabular-nums text-[#9AA1AD]">
                        ({t("forecast.uncertaintyRange")}: {Number(p10).toFixed(0)}–{Number(p90).toFixed(0)}%)
                      </span>
                    )}
                  </div>
                  {fovExit != null && Number(fovExit) >= 0.35 && (
                    <div className="text-[10px] font-mono text-amber-300/80 mt-0.5">
                      {t("forecast.fovExit")}: {(Number(fovExit) * 100).toFixed(0)}%
                    </div>
                  )}
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
            );
          })}
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

          {forecasts.length > 0 && forecasts[0]?.status === "available" && (
            <div className="pt-3 border-t border-white/[0.06] space-y-2">
              <div className="flex items-center justify-between gap-2">
                <span className="obs-label flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-sky-400" /> {t("forecast.advectedPreview")}
                </span>
                <div className="flex items-center gap-1">
                  {[5, 15, 20].map((h) => (
                    <button
                      key={h}
                      type="button"
                      onClick={() => {
                        setAdvHorizon(h);
                        setAdvErr(false);
                      }}
                      className={`px-2 py-0.5 rounded text-[10px] font-mono transition ${
                        advHorizon === h
                          ? "bg-sky-500/20 text-sky-300 border border-sky-500/40"
                          : "bg-white/[0.05] text-[#8E95A2] hover:text-white"
                      }`}
                    >
                      +{h}m
                    </button>
                  ))}
                </div>
              </div>
              <div className="relative aspect-video w-full rounded-lg overflow-hidden bg-[#08090B] border border-white/[0.08]">
                <img
                  src={`/api/camera/latest?layer=raw&t=${advNonce}`}
                  alt="Sky background frame"
                  className="w-full h-full object-contain opacity-55 select-none pointer-events-none"
                />
                {!advErr && (
                  <img
                    src={`/api/forecast/advected-mask.png?horizon=${advHorizon}&t=${advNonce}`}
                    alt="Advected cloud probability map"
                    className="absolute inset-0 w-full h-full object-contain transition-opacity duration-200 pointer-events-none"
                    onLoad={(e) => {
                      (e.currentTarget as HTMLImageElement).style.opacity = "0.88";
                    }}
                    onError={() => setAdvErr(true)}
                  />
                )}
                <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/70 border border-white/10 text-[10px] font-mono text-sky-200">
                  +{advHorizon}m ·{" "}
                  {selectedAdvFc?.baselines?.optical_flow_cover_pct != null
                    ? `${Number(selectedAdvFc.baselines.optical_flow_cover_pct).toFixed(0)}% OF`
                    : `${selectedAdvFc?.cloud_cover_prediction?.toFixed(0) ?? "—"}%`}
                  {selectedAdvFc?.baselines?.fov_exit_fraction != null
                    ? ` · FOV exit ${(Number(selectedAdvFc.baselines.fov_exit_fraction) * 100).toFixed(0)}%`
                    : ""}
                </div>
              </div>
              <div className="flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono text-[#8E95A2]">
                <div className="flex items-center gap-2">
                  <span>{isIt ? "Prob. nube:" : "Cloud prob:"}</span>
                  <span className="inline-flex items-center gap-1">
                    <span className="w-2.5 h-2.5 rounded-sm bg-[#38BDF8]" /> 20%
                  </span>
                  <span className="inline-flex items-center gap-1">
                    <span className="w-2.5 h-2.5 rounded-sm bg-[#F8FAFC]" /> 50%
                  </span>
                  <span className="inline-flex items-center gap-1">
                    <span className="w-2.5 h-2.5 rounded-sm bg-[#C378EB]" /> 100%
                  </span>
                </div>
                {selectedAdvFc?.cover_p10 != null && selectedAdvFc?.cover_p90 != null && (
                  <span>
                    p10–p90: {Number(selectedAdvFc.cover_p10).toFixed(0)}–{Number(selectedAdvFc.cover_p90).toFixed(0)}%
                  </span>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="obs-panel p-5 space-y-3">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" /> {t("forecast.verificationTitle")}
          </h3>
          {!verification || verification.total_verified === 0 ? (
            <p className="text-xs text-[#8E95A2]">{t("forecast.verificationEmpty")}</p>
          ) : (
            <div className="space-y-3">
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2 text-xs">
                <div className="obs-inset p-2.5">
                  <div className="obs-label">{t("forecast.verified")}</div>
                  <div className="font-mono tabular-nums font-bold text-white mt-0.5">{verification.total_verified}</div>
                </div>
                <div className="obs-inset p-2.5">
                  <div className="obs-label">{t("forecast.mae")}</div>
                  <div className="font-mono tabular-nums font-bold text-sky-300 mt-0.5">{verification.overall.mae_pct}%</div>
                </div>
                <div className="obs-inset p-2.5">
                  <div className="obs-label">{t("forecast.persMae")}</div>
                  <div className="font-mono tabular-nums font-bold text-[#C5CBD5] mt-0.5">
                    {verification.overall.persistence_mae_pct != null ? `${verification.overall.persistence_mae_pct}%` : "—"}
                  </div>
                </div>
                <div className="obs-inset p-2.5">
                  <div className="obs-label">{t("forecast.skillVsPers")}</div>
                  <div
                    className={`font-mono tabular-nums font-bold mt-0.5 ${
                      verification.overall.beats_persistence ? "text-emerald-300" : "text-amber-300"
                    }`}
                  >
                    {formatSkill(verification.overall.skill_vs_persistence)}
                  </div>
                </div>
                <div className="obs-inset p-2.5">
                  <div className="obs-label">{t("forecast.rmse")}</div>
                  <div className="font-mono tabular-nums font-bold text-sky-300 mt-0.5">{verification.overall.rmse_pct}%</div>
                </div>
                <div className="obs-inset p-2.5">
                  <div className="obs-label">{t("forecast.ofMae")}</div>
                  <div className="font-mono tabular-nums font-bold text-[#C5CBD5] mt-0.5">
                    {verification.overall.optical_flow_mae_pct != null ? `${verification.overall.optical_flow_mae_pct}%` : "—"}
                  </div>
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

              {verification.overall.persistence_mae_pct === 0 && (
                <div className="px-3 py-2 rounded-lg bg-amber-500/10 border border-amber-400/25 text-[11px] text-amber-200/90">
                  {isIt
                    ? "Cielo rimasto statico durante queste verifiche (MAE persistenza = 0%): lo skill relativo alla persistenza è N/D finché la copertura nuvolosa non varia."
                    : "Sky remained static across verified samples (persistence MAE = 0%): skill vs persistence is N/A until cloud cover changes."}
                </div>
              )}

              {verification.by_horizon && Object.keys(verification.by_horizon).length > 0 && (
                <div>
                  <div className="obs-label mb-1.5">{t("forecast.byHorizon")}</div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-[11px] font-mono tabular-nums text-left">
                      <thead className="text-[#8E95A2] border-b border-white/[0.08]">
                        <tr>
                          <th className="py-1 pr-2">+min</th>
                          <th className="py-1 pr-2">n</th>
                          <th className="py-1 pr-2">{t("forecast.mae")}</th>
                          <th className="py-1 pr-2">{t("forecast.persMae")}</th>
                          <th className="py-1 pr-2">{t("forecast.ofMae")}</th>
                          <th className="py-1">{t("forecast.skillVsPers")}</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/[0.05]">
                        {Object.entries(verification.by_horizon).map(([h, stats]: [string, any]) => (
                          <tr key={h}>
                            <td className="py-1 pr-2 text-sky-300">+{h}m</td>
                            <td className="py-1 pr-2 text-[#8E95A2]">{stats.count}</td>
                            <td className="py-1 pr-2 text-white">{stats.mae_pct}%</td>
                            <td className="py-1 pr-2 text-[#C5CBD5]">
                              {stats.persistence_mae_pct != null ? `${stats.persistence_mae_pct}%` : "—"}
                            </td>
                            <td className="py-1 pr-2 text-[#C5CBD5]">
                              {stats.optical_flow_mae_pct != null ? `${stats.optical_flow_mae_pct}%` : "—"}
                            </td>
                            <td className={`py-1 ${stats.beats_persistence ? "text-emerald-300" : "text-amber-300"}`}>
                              {formatSkill(stats.skill_vs_persistence)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

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
