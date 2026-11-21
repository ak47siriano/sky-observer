import React from "react";
import { AlertTriangle, Camera, Cloud, Compass, Eye, Gauge, RefreshCw, Sun, Upload, Wind } from "lucide-react";
import { ProbabilityBars } from "../components/Charts";
import { useI18n } from "../i18n/I18nContext";
import type { CameraStatus, EventAlert, ForecastItem, Observation } from "../types/api";

interface Props {
  camera: CameraStatus | null;
  obs: Observation | null;
  forecasts: ForecastItem[];
  events: EventAlert[];
  onlineWeather?: any;
  offlineMode?: boolean;
  onTriggerSnapshot: () => void;
  onUploadFile: (file: File) => void;
  onNavigate: (page: string) => void;
  busy: boolean;
}

export const OverviewPage: React.FC<Props> = ({
  camera,
  obs,
  forecasts,
  events,
  onlineWeather,
  offlineMode = true,
  onTriggerSnapshot,
  onUploadFile,
  onNavigate,
  busy,
}) => {
  const { t, tEnum, tCardinal } = useI18n();
  const next15 = forecasts.find((f) => f.horizon_minutes === 15) || forecasts[0];
  const imgSrc = obs ? `/api/analysis/image/${obs.id}?layer=raw&t=${encodeURIComponent(obs.captured_at)}` : null;

  return (
    <div className="space-y-5">
      {camera && !camera.connected && !obs && (
        <div className="bg-amber-950/35 border border-amber-700/60 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-400 mt-0.5 shrink-0" />
            <div>
              <p className="text-sm font-semibold text-amber-200">
                {t("overview.camNotConnected", { type: camera.source_type })}
              </p>
              <p className="text-xs text-amber-300/80 mt-0.5">
                {camera.last_error || t("overview.camNotConnectedHint")}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <label className="cursor-pointer inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 border border-slate-700">
              <Upload className="w-3.5 h-3.5" />
              {t("overview.importFile")}
              <input
                type="file"
                accept="image/*,video/*"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && onUploadFile(e.target.files[0])}
              />
            </label>
            <button
              onClick={() => onNavigate("settings")}
              className="px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 text-xs font-medium text-white"
            >
              {t("overview.configureCam")}
            </button>
          </div>
        </div>
      )}

      {camera?.is_demo_mode && (
        <div className="bg-sky-950/40 border border-sky-700/60 rounded-xl px-4 py-2.5 flex items-center justify-between text-xs text-sky-200">
          <span>{t("overview.demoBanner")}</span>
        </div>
      )}

      {/* KPI Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-[#0F1623] border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>{t("overview.cloudCoverage")}</span>
            <Cloud className="w-4 h-4 text-sky-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-white">
              {obs ? `${obs.cloud_coverage_pct.toFixed(1)}%` : "—"}
            </span>
            <span className="text-xs text-slate-400">
              {obs ? t("overview.clearPct", { pct: obs.clear_sky_pct.toFixed(1) }) : t("overview.noFrame")}
            </span>
          </div>
          <div className="mt-1.5 text-xs text-sky-400">
            {t("overview.trend")}: {tEnum(obs?.coverage?.trend || "unavailable")}
          </div>
        </div>

        <div className="bg-[#0F1623] border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>{t("overview.dominantCloudType")}</span>
            <Eye className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="mt-2 text-xl font-bold text-white truncate">
            {obs ? tEnum(obs.dominant_cloud_type) : t("overview.unavailable")}
          </div>
          <div className="mt-1.5 text-xs text-slate-400">
            {t("overview.visualConf")}: {obs ? `${(obs.classification_confidence * 100).toFixed(0)}%` : "—"}
          </div>
        </div>

        <div className="bg-[#0F1623] border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>{t("overview.cloudAdvection")}</span>
            <Wind className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-white">
              {obs?.motion?.available ? `${obs.motion_speed_px_s.toFixed(1)} px/s` : "—"}
            </span>
            {obs?.motion_speed_deg_s != null && (
              <span className="text-xs font-mono text-cyan-300">
                ({obs.motion_speed_deg_s.toFixed(2)}°/s)
              </span>
            )}
          </div>
          <div className="mt-1.5 text-xs text-slate-400 flex items-center gap-1.5">
            <Compass className="w-3.5 h-3.5 text-cyan-400" />
            {obs?.motion?.available
              ? `${tCardinal(obs.motion_direction_cardinal)} (${obs.motion_direction_deg.toFixed(0)}°) · conf ${(obs.motion_confidence * 100).toFixed(0)}%`
              : t("overview.needSequential")}
          </div>
        </div>

        <div className="bg-[#0F1623] border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>{t("overview.skyBrightness")}</span>
            <Sun className="w-4 h-4 text-amber-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-white">
              {obs ? `${obs.avg_luminance.toFixed(1)} L*` : "—"}
            </span>
            <span className="text-xs font-mono text-slate-400">
              BSI {obs ? obs.blue_sky_index.toFixed(2) : "—"}
            </span>
          </div>
          <div className="mt-1.5 text-xs text-amber-300">
            {obs ? tEnum(obs.visual_condition) : t("overview.noObservation")}
          </div>
        </div>
      </div>

      {/* Main Feed + Forecast + Alerts */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
        <div className="xl:col-span-2 bg-[#0F1623] border border-slate-800 rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-800 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2.5">
              <Camera className="w-4 h-4 text-sky-400" />
              <span className="text-sm font-semibold text-white">{camera?.camera_name || "Sky Camera"}</span>
              <span
                className={`px-2 py-0.5 rounded-full text-[11px] font-medium ${
                  camera?.connected
                    ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
                    : "bg-rose-500/15 text-rose-300 border border-rose-500/30"
                }`}
              >
                {camera?.connected ? t("overview.connected") : t("overview.disconnected")}
              </span>
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <span>
                {t("overview.last")}: {obs ? new Date(obs.captured_at).toLocaleTimeString() : t("overview.never")}
              </span>
              <button
                onClick={onTriggerSnapshot}
                disabled={busy}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 disabled:opacity-50"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${busy ? "animate-spin" : ""}`} />
                {t("overview.captureNow")}
              </button>
              <label className="cursor-pointer inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-sky-600/20 hover:bg-sky-600/30 text-sky-300 border border-sky-500/30">
                <Upload className="w-3.5 h-3.5" />
                {t("overview.upload")}
                <input
                  type="file"
                  accept="image/*,video/*"
                  className="hidden"
                  onChange={(e) => e.target.files?.[0] && onUploadFile(e.target.files[0])}
                />
              </label>
            </div>
          </div>

          <div className="aspect-video bg-[#060911] flex items-center justify-center relative">
            {imgSrc ? (
              <img src={imgSrc} alt="Latest sky capture" className="w-full h-full object-contain" />
            ) : (
              <div className="text-center space-y-2 p-6">
                <Camera className="w-10 h-10 text-slate-600 mx-auto" />
                <p className="text-sm text-slate-400">{t("overview.noImageYet")}</p>
                <p className="text-xs text-slate-500">{t("overview.connectOrUpload")}</p>
              </div>
            )}
            {obs && (
              <div className="absolute bottom-3 left-3 right-3 flex flex-wrap justify-between gap-2 pointer-events-none">
                <div className="bg-slate-950/80 backdrop-blur border border-slate-800 rounded-lg px-2.5 py-1 text-[11px] font-mono text-slate-200">
                  {t("overview.quality")}: {(obs.quality_score * 100).toFixed(0)}% · {obs.width}×{obs.height} ·{" "}
                  {t("overview.inference")} {obs.inference_duration_ms.toFixed(0)}ms
                </div>
                <div className="bg-slate-950/80 backdrop-blur border border-slate-800 rounded-lg px-2.5 py-1 text-[11px] font-mono text-sky-300">
                  {t("overview.thin")} {obs.thin_cloud_pct.toFixed(0)}% · {t("overview.dense")}{" "}
                  {obs.dense_cloud_pct.toFixed(0)}% · {t("overview.dark")} {obs.dark_cloud_pct.toFixed(0)}%
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="space-y-4">
          <div className="bg-[#0F1623] border border-slate-800 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Gauge className="w-4 h-4 text-sky-400" />
                <h3 className="text-sm font-semibold text-white">{t("overview.shortTermNowcast")}</h3>
              </div>
              <button onClick={() => onNavigate("forecast")} className="text-xs text-sky-400 hover:underline">
                {t("overview.details")}
              </button>
            </div>

            {next15 && next15.status === "available" ? (
              <>
                <div className="flex items-baseline justify-between bg-slate-900/70 rounded-lg p-3 border border-slate-800">
                  <div>
                    <div className="text-xs text-slate-400">{t("overview.predictedCondition")}</div>
                    <div className="text-base font-semibold text-white mt-0.5">
                      {tEnum(next15.forecast)}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs text-slate-400">{t("overview.cloudCover")}</div>
                    <div className="text-xl font-bold font-mono text-sky-400">
                      {next15.cloud_cover_prediction.toFixed(0)}%
                    </div>
                  </div>
                </div>
                <ProbabilityBars
                  items={[
                    { label: t("overview.clearingLikelihood"), value: next15.clearing_prob, color: "#34D399" },
                    { label: t("overview.increasingCloudiness"), value: next15.increasing_prob, color: "#38BDF8" },
                    { label: t("overview.overcastLikelihood"), value: next15.overcast_prob, color: "#94A3B8" },
                    { label: t("overview.forecastConfidence"), value: next15.confidence, color: "#FBBF24" },
                  ]}
                />
                <div className="text-[11px] text-slate-400 border-t border-slate-800 pt-2">
                  {t("overview.imageOnlyModel")} <span className="font-mono">{next15.model_version}</span>
                </div>
              </>
            ) : (
              <div className="text-xs text-slate-400 py-6 text-center">{t("overview.forecastUnavailable")}</div>
            )}
          </div>

          <div className="bg-[#0F1623] border border-slate-800 rounded-xl p-4 space-y-2.5">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white">{t("overview.recentAlerts")}</h3>
              <button onClick={() => onNavigate("events")} className="text-xs text-sky-400 hover:underline">
                {t("overview.viewAll")}
              </button>
            </div>
            {(() => {
              const visibleEvents = events.filter(
                (ev) => (ev.user_status ?? (ev.acknowledged ? "confirmed" : "pending")) !== "rejected"
              );
              return visibleEvents.length === 0 ? (
                <p className="text-xs text-slate-500 py-3">{t("overview.noAlerts")}</p>
              ) : (
                <div className="space-y-2 max-h-52 overflow-y-auto pr-1">
                  {visibleEvents.slice(0, 5).map((ev) => (
                    <div key={ev.id} className="p-2.5 rounded-lg bg-slate-900/70 border border-slate-800/80 text-xs">
                      <div className="flex items-center justify-between gap-2">
                        <span
                          className={`font-semibold ${
                            ev.severity === "critical"
                              ? "text-rose-400"
                              : ev.severity === "warning"
                              ? "text-amber-300"
                              : "text-sky-300"
                          }`}
                        >
                          {ev.title}
                        </span>
                        <span className="text-[10px] font-mono text-slate-500">
                          {new Date(ev.created_at).toLocaleTimeString()}
                        </span>
                      </div>
                      <p className="text-slate-400 mt-0.5">{ev.description}</p>
                    </div>
                  ))}
                </div>
              );
            })()}
          </div>

          {!offlineMode && onlineWeather?.available && onlineWeather.current && (
            <div className="bg-[#0F1623] border border-sky-800/60 rounded-xl p-4 space-y-2.5">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-sky-200">{t("radar.onlineCard")}</h3>
                <button onClick={() => onNavigate("radar")} className="text-xs text-sky-400 hover:underline">
                  {t("nav.radar")} →
                </button>
              </div>
              <div className="grid grid-cols-3 gap-2 text-xs font-mono">
                <div className="bg-slate-900/80 rounded p-2">
                  <div className="text-[10px] text-slate-400">{t("wx.temp")}</div>
                  <div className="text-white font-bold">{onlineWeather.current.temperature_c}°C</div>
                </div>
                <div className="bg-slate-900/80 rounded p-2">
                  <div className="text-[10px] text-slate-400">{t("wx.humidity")}</div>
                  <div className="text-white font-bold">{onlineWeather.current.relative_humidity_pct}%</div>
                </div>
                <div className="bg-slate-900/80 rounded p-2">
                  <div className="text-[10px] text-slate-400">{t("wx.pressure")}</div>
                  <div className="text-white font-bold">{onlineWeather.current.pressure_msl_hpa} hPa</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
