import React, { useState } from "react";
import { AlertTriangle, Check, Navigation, Upload, X } from "lucide-react";
import { ProbabilityBars } from "../components/Charts";
import { SkyStage } from "../components/SkyStage";
import { useI18n } from "../i18n/I18nContext";
import { api } from "../services/api";
import type { CameraStatus, EventAlert, ForecastItem, Observation } from "../types/api";

interface DashboardPageProps {
  camera: CameraStatus | null;
  obs: Observation | null;
  forecasts: ForecastItem[];
  events: EventAlert[];
  onlineWeather?: any;
  offlineMode: boolean;
  onTriggerSnapshot: () => void;
  onUploadFile: (file: File) => void;
  onRefreshAll: () => void;
  onNavigate: (page: string) => void;
  busy: boolean;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({
  camera,
  obs,
  forecasts,
  events,
  onlineWeather,
  offlineMode,
  onTriggerSnapshot,
  onUploadFile,
  onRefreshAll,
  onNavigate,
  busy,
}) => {
  const { t, tEnum, tCardinal } = useI18n();
  const [selectedHorizon, setSelectedHorizon] = useState<number>(15);
  const [alertBusyId, setAlertBusyId] = useState<number | null>(null);
  const [previewAlertObsId, setPreviewAlertObsId] = useState<number | null>(null);

  const selectedForecast =
    forecasts.find((f) => f.horizon_minutes === selectedHorizon) ||
    forecasts.find((f) => f.horizon_minutes === 15) ||
    forecasts[0];

  const visibleEvents = events.filter(
    (ev) => (ev.user_status ?? (ev.acknowledged ? "confirmed" : "pending")) !== "rejected"
  );

  const quickFeedback = async (id: number, status: "confirmed" | "rejected") => {
    setAlertBusyId(id);
    try {
      await api.setEventFeedback(id, status);
      onRefreshAll();
    } finally {
      setAlertBusyId(null);
    }
  };

  return (
    <div className="space-y-4">
      {camera && !camera.connected && !obs && (
        <div className="obs-panel border-amber-500/30 bg-amber-950/20 px-4 py-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2.5 text-xs">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
            <div>
              <span className="font-semibold text-amber-200">
                {t("overview.camNotConnected", { type: camera.source_type })}
              </span>
              <span className="text-amber-300/75 ml-2">
                {camera.last_error || t("overview.camNotConnectedHint")}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <label className="obs-btn-secondary cursor-pointer inline-flex items-center gap-1.5 px-3 py-1.5">
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
              type="button"
              onClick={() => onNavigate("settings")}
              className="obs-btn-primary px-3 py-1.5"
            >
              {t("overview.configureCam")}
            </button>
          </div>
        </div>
      )}

      {camera?.is_demo_mode && (
        <div className="obs-inset px-4 py-2 text-xs text-[#9AA1AD]">
          {t("overview.demoBanner")}
        </div>
      )}

      {/* Meteorological Station METAR / SYNOP Bar */}
      {obs && (
        <div className="obs-panel bg-[#15181E] px-3.5 py-2 flex flex-wrap items-center justify-between gap-x-5 gap-y-1 text-[11px] font-mono tabular-nums">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
            <span className="inline-flex items-center gap-1.5 text-[#ECEFF4] font-semibold">
              <span className={`w-1.5 h-1.5 rounded-full ${camera?.connected ? "bg-emerald-500" : "bg-rose-500"}`} />
              {onlineWeather?.location_name || camera?.location_description || "LOCAL STATION"}
            </span>
            <span className="text-[#60A5FA] font-semibold">
              {obs.cloud_coverage_pct < 6
                ? "SKC"
                : obs.cloud_coverage_pct < 31
                ? "FEW"
                : obs.cloud_coverage_pct < 56
                ? "SCT"
                : obs.cloud_coverage_pct < 88
                ? "BKN"
                : "OVC"}{" "}
              {Math.round((obs.cloud_coverage_pct / 100) * 8)}/8 ({obs.cloud_coverage_pct.toFixed(0)}%)
            </span>
            <span className="text-[#ECEFF4]">{tEnum(obs.dominant_cloud_type)}</span>
            {obs.motion?.available && (
              <span className="text-[#8D96A5]">
                ADV {tCardinal(obs.motion_direction_cardinal)} {obs.motion_direction_deg.toFixed(0)}° @{" "}
                {obs.motion_speed_px_s.toFixed(1)} px/s
              </span>
            )}
          </div>
          {!offlineMode && onlineWeather?.available && onlineWeather.current && (
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[#8D96A5]">
              <span>
                T/Td{" "}
                <strong className="text-[#ECEFF4]">
                  {onlineWeather.current.temperature_c}°/{onlineWeather.current.dewpoint_c ?? "—"}°
                </strong>
              </span>
              <span>
                QNH <strong className="text-[#ECEFF4]">{onlineWeather.current.pressure_msl_hpa} hPa</strong>
              </span>
              {onlineWeather.current.estimated_cloud_base_m != null && (
                <span>
                  LCL <strong className="text-[#ECEFF4]">~{onlineWeather.current.estimated_cloud_base_m} m</strong>
                </span>
              )}
            </div>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-3 xl:h-[calc(100vh-112px)] xl:min-h-[600px]">
        {/* Left/Center: Full-Height Sky Monitor */}
        <div className="xl:col-span-8 xl:h-full min-h-0 flex flex-col">
          <SkyStage
            camera={camera}
            obs={obs}
            onTriggerSnapshot={onTriggerSnapshot}
            onUploadFile={onUploadFile}
            onMaskSaved={onRefreshAll}
            busy={busy}
          />
        </div>

        {/* Right: MeteoBlue-Style Unified Docked Telemetry & Meteogram Column */}
        <div className="xl:col-span-4 xl:h-full obs-panel divide-y divide-[#262B34] overflow-y-auto flex flex-col">
          {/* 1. Current Sky State */}
          <div className="p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="obs-label">{t("dash.skyState")}</span>
              <span className="inline-flex items-center gap-1.5 text-[11px] font-mono text-[#9AA1AD]">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    camera?.connected ? "bg-emerald-400" : "bg-rose-400"
                  }`}
                />
                {camera?.connected ? t("overview.connected") : t("overview.disconnected")}
              </span>
            </div>

            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-3.5">
                {/* Radial Cloud Coverage Gauge */}
                <div className="relative w-16 h-16 shrink-0">
                  <svg viewBox="0 0 64 64" className="w-full h-full -rotate-90">
                    <circle
                      cx="32"
                      cy="32"
                      r="26"
                      fill="none"
                      stroke="rgba(255,255,255,0.08)"
                      strokeWidth="6"
                    />
                    <circle
                      cx="32"
                      cy="32"
                      r="26"
                      fill="none"
                      stroke="url(#skyGaugeGrad)"
                      strokeWidth="6"
                      strokeLinecap="round"
                      strokeDasharray={2 * Math.PI * 26}
                      strokeDashoffset={
                        2 * Math.PI * 26 * (1 - Math.min(100, Math.max(0, obs?.cloud_coverage_pct ?? 0)) / 100)
                      }
                      className="transition-all duration-700"
                    />
                    <defs>
                      <linearGradient id="skyGaugeGrad" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0%" stopColor="#38BDF8" />
                        <stop offset="100%" stopColor="#818CF8" />
                      </linearGradient>
                    </defs>
                  </svg>
                  <div className="absolute inset-0 flex flex-col items-center justify-center font-mono tabular-nums">
                    <span className="text-sm font-bold text-white leading-none">
                      {obs ? `${obs.cloud_coverage_pct.toFixed(0)}%` : "—"}
                    </span>
                    {obs && (
                      <span className="text-[9px] text-sky-300/90 mt-0.5">
                        {Math.round((obs.cloud_coverage_pct / 100) * 8)}/8
                      </span>
                    )}
                  </div>
                </div>
                <div>
                  <div className="text-lg font-semibold tracking-tight text-white leading-snug flex items-center gap-2">
                    <span>{obs ? tEnum(obs.visual_condition) : t("overview.noObservation")}</span>
                    {obs && (
                      <span className="px-1.5 py-0.5 rounded bg-white/[0.06] border border-white/[0.1] font-mono text-[10px] text-sky-200">
                        {obs.cloud_coverage_pct < 6
                          ? "SKC"
                          : obs.cloud_coverage_pct < 31
                          ? "FEW"
                          : obs.cloud_coverage_pct < 56
                          ? "SCT"
                          : obs.cloud_coverage_pct < 88
                          ? "BKN"
                          : "OVC"}
                      </span>
                    )}
                  </div>
                  {obs && (
                    <div className="text-xs font-medium text-sky-300 mt-0.5 flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-sky-400" />
                      {tEnum(obs.coverage?.trend || "unavailable")}
                    </div>
                  )}
                </div>
              </div>
              {obs && (
                <div className="text-right shrink-0">
                  <div className="obs-label">{t("overview.clearPct", { pct: obs.clear_sky_pct.toFixed(0) })}</div>
                  <div className="text-[11px] font-mono tabular-nums text-[#8A96AA] mt-0.5">
                    {obs.inference_duration_ms.toFixed(0)} ms
                  </div>
                </div>
              )}
            </div>

            {/* Cloud Layer Composition Bar */}
            {obs && (
              <div className="space-y-2">
                <div className="h-2 rounded-full overflow-hidden flex bg-white/[0.06]">
                  <div
                    style={{ width: `${Math.max(0, obs.clear_sky_pct)}%` }}
                    className="bg-[#2563EB]"
                    title={t("vision.legend.clear")}
                  />
                  <div
                    style={{ width: `${Math.max(0, obs.thin_cloud_pct)}%` }}
                    className="bg-[#38BDF8]"
                    title={t("vision.legend.thin")}
                  />
                  <div
                    style={{ width: `${Math.max(0, obs.dense_cloud_pct)}%` }}
                    className="bg-[#F8FAFC]"
                    title={t("vision.legend.dense")}
                  />
                  <div
                    style={{ width: `${Math.max(0, obs.dark_cloud_pct)}%` }}
                    className="bg-[#C34BEB]"
                    title={t("vision.legend.dark")}
                  />
                  <div
                    style={{ width: `${Math.max(0, obs.sun_glare_pct ?? obs.haze_fog_pct ?? 0)}%` }}
                    className="bg-[#FBBF24]"
                    title={t("vision.legend.sunGlare")}
                  />
                </div>
                <div className="flex flex-wrap justify-between text-[11px] font-mono tabular-nums text-[#8E95A2]">
                  <span>{t("overview.clearPct", { pct: obs.clear_sky_pct.toFixed(0) })}</span>
                  <span>
                    {t("overview.thin")} {obs.thin_cloud_pct.toFixed(0)}% · {t("overview.dense")}{" "}
                    {obs.dense_cloud_pct.toFixed(0)}% · {t("overview.dark")} {obs.dark_cloud_pct.toFixed(0)}%
                  </span>
                </div>
              </div>
            )}

            {/* Hairline 3-Metric Table */}
            <div className="grid grid-cols-3 divide-x divide-[#262B34] border border-[#262B34] rounded">
              <div className="p-2">
                <div className="obs-label">{t("overview.dominantCloudType")}</div>
                <div className="text-xs font-semibold text-[#ECEFF4] truncate mt-1">
                  {obs ? tEnum(obs.dominant_cloud_type) : "—"}
                </div>
                <div className="text-[11px] font-mono tabular-nums text-[#8D96A5]">
                  {obs ? `${(obs.classification_confidence * 100).toFixed(0)}% conf` : "—"}
                </div>
              </div>

              <div className="p-2">
                <div className="obs-label">{t("dash.advection")}</div>
                <div className="flex items-center gap-1 mt-1">
                  {obs?.motion?.available && (
                    <Navigation
                      className="w-3 h-3 text-[#8D96A5] shrink-0"
                      style={{ transform: `rotate(${obs.motion_direction_deg}deg)` }}
                    />
                  )}
                  <span className="text-xs font-mono tabular-nums font-semibold text-[#ECEFF4]">
                    {obs?.motion?.available ? `${obs.motion_speed_px_s.toFixed(1)} px/s` : "—"}
                  </span>
                </div>
                <div className="text-[11px] font-mono tabular-nums text-[#8D96A5]">
                  {obs?.motion?.available
                    ? `${tCardinal(obs.motion_direction_cardinal)} ${obs.motion_direction_deg.toFixed(0)}°`
                    : "—"}
                </div>
              </div>

              <div className="p-2">
                <div className="obs-label">{t("dash.luminance")}</div>
                <div className="text-xs font-mono tabular-nums font-semibold text-[#ECEFF4] mt-1">
                  {obs ? `${obs.avg_luminance.toFixed(0)} L*` : "—"}
                </div>
                <div className="text-[11px] font-mono tabular-nums text-[#8D96A5]">
                  {t("dash.quality")} {obs ? `${(obs.quality_score * 100).toFixed(0)}%` : "—"}
                </div>
              </div>
            </div>

            {!offlineMode && onlineWeather?.available && onlineWeather.current && (
              <div className="pt-3 border-t border-white/[0.07] space-y-2">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5 min-w-0">
                    <span className="obs-label shrink-0">{t("dash.localWeather")}</span>
                    {(onlineWeather.location_name || onlineWeather.latitude != null) && (
                      <span className="text-[11px] font-medium text-sky-300 truncate">
                        ·{" "}
                        {onlineWeather.location_name ||
                          `${Number(onlineWeather.latitude).toFixed(3)}°, ${Number(onlineWeather.longitude).toFixed(3)}°`}
                      </span>
                    )}
                  </div>
                  <button
                    type="button"
                    onClick={() => onNavigate("radar")}
                    className="text-[11px] text-[#9AA1AD] hover:text-white transition shrink-0"
                  >
                    {t("dash.openMap")} →
                  </button>
                </div>
                <div className="grid grid-cols-4 divide-x divide-[#262B34] border border-[#262B34] rounded text-center font-mono tabular-nums text-xs">
                  <div className="py-1.5 px-1">
                    <div className="text-[10px] text-[#8D96A5]">{t("wx.temp")}</div>
                    <div className="text-[#ECEFF4] font-semibold">{onlineWeather.current.temperature_c}°</div>
                  </div>
                  <div className="py-1.5 px-1">
                    <div className="text-[10px] text-[#8D96A5]">{t("wx.humidity")}</div>
                    <div className="text-[#ECEFF4] font-semibold">{onlineWeather.current.relative_humidity_pct}%</div>
                  </div>
                  <div className="py-1.5 px-1">
                    <div className="text-[10px] text-[#8D96A5]">{t("wx.pressure")}</div>
                    <div className="text-[#ECEFF4] font-semibold">{onlineWeather.current.pressure_msl_hpa}</div>
                  </div>
                  <div className="py-1.5 px-1">
                    <div className="text-[10px] text-[#8D96A5]">{t("wx.wind")}</div>
                    <div className="text-[#ECEFF4] font-semibold">{onlineWeather.current.wind_speed_kmh}</div>
                  </div>
                </div>
                {(onlineWeather.current.dewpoint_c != null || onlineWeather.current.estimated_cloud_base_m != null) && (
                  <div className="flex items-center justify-between text-[11px] font-mono text-[#9AA1AD] px-1">
                    {onlineWeather.current.dewpoint_c != null && (
                      <span>
                        {t("radar.dewpoint")}: <strong className="text-white">{onlineWeather.current.dewpoint_c}°C</strong>
                      </span>
                    )}
                    {onlineWeather.current.estimated_cloud_base_m != null && (
                      <span>
                        {t("radar.cloudBaseLcl")}:{" "}
                        <strong className="text-sky-300">~{onlineWeather.current.estimated_cloud_base_m} m</strong>
                      </span>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* 2. Short-Term Nowcast Meteogram Block */}
          <div className="p-4 space-y-3">
            <div className="flex items-center justify-between gap-2">
              <span className="obs-label">{t("dash.nowcast")}</span>
              <span className="text-[10px] font-mono text-[#64748B]">METEOGRAM</span>
            </div>

            {forecasts.length > 0 && (
              <div className="grid grid-cols-4 divide-x divide-[#262B34] border border-[#262B34] rounded overflow-hidden">
                {forecasts.slice(0, 4).map((f) => {
                  const active = selectedForecast?.horizon_minutes === f.horizon_minutes;
                  return (
                    <button
                      key={f.horizon_minutes}
                      type="button"
                      onClick={() => setSelectedHorizon(f.horizon_minutes)}
                      className={`p-2 text-center transition-colors ${
                        active
                          ? "bg-[#1F2632] shadow-[inset_0_-2px_0_0_#3B82F6]"
                          : "bg-[#16191F] hover:bg-[#1B1F27]"
                      }`}
                    >
                      <div className="text-[10px] font-mono text-[#8D96A5]">+{f.horizon_minutes}m</div>
                      <div className={`text-sm font-mono tabular-nums font-bold mt-0.5 ${active ? "text-[#ECEFF4]" : "text-[#CBD5E1]"}`}>
                        {f.cloud_cover_prediction.toFixed(0)}%
                      </div>
                      <div className="text-[9px] font-mono text-[#5D6675]">
                        {Math.round((f.cloud_cover_prediction / 100) * 8)}/8
                      </div>
                    </button>
                  );
                })}
              </div>
            )}

            {selectedForecast && selectedForecast.status === "available" ? (
              <>
                <div className="flex items-baseline justify-between obs-inset p-3">
                  <div>
                    <div className="text-sm font-semibold text-white">{tEnum(selectedForecast.forecast)}</div>
                    <div className="text-[11px] font-mono tabular-nums text-[#8E95A2] mt-0.5">
                      {t("overview.forecastConfidence")} {(selectedForecast.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="obs-label">{t("overview.cloudCover")}</div>
                    <div className="text-2xl font-semibold font-mono tabular-nums text-sky-300">
                      {selectedForecast.cloud_cover_prediction.toFixed(0)}%
                    </div>
                  </div>
                </div>
                <ProbabilityBars
                  items={[
                    { label: t("overview.clearingLikelihood"), value: selectedForecast.clearing_prob, color: "#34D399" },
                    { label: t("overview.increasingCloudiness"), value: selectedForecast.increasing_prob, color: "#60A5FA" },
                    { label: t("overview.overcastLikelihood"), value: selectedForecast.overcast_prob, color: "#94A3B8" },
                  ]}
                />

                {/* Plain-language "Why This Forecast?" Explainability Box */}
                {obs && (
                  <div className="obs-inset p-3 space-y-1.5 text-[11px]">
                    <div className="font-semibold text-sky-300">{t("dash.whyForecastTitle")}</div>
                    <ul className="space-y-1 text-[#C5CBD5] list-disc list-inside">
                      <li>
                        {t("dash.whyCoverTrend", {
                          cov: obs.cloud_coverage_pct.toFixed(0),
                          trend: tEnum(obs.coverage?.trend || "stable"),
                          rate: Number(obs.coverage?.rate_of_change_pct_per_min ?? 0).toFixed(2),
                        })}
                      </li>
                      {obs.motion?.available && obs.motion_speed_px_s > 0.15 && (
                        <li>
                          {t("dash.whyAdvection", {
                            dir: tCardinal(obs.motion_direction_cardinal),
                            deg: obs.motion_direction_deg.toFixed(0),
                            px: obs.motion_speed_px_s.toFixed(1),
                          })}
                        </li>
                      )}
                      {Number(obs.color?.luminance_delta ?? 0) < -4 && (
                        <li className="text-amber-300">
                          {t("dash.whyDarkening", {
                            delta: Number(obs.color?.luminance_delta ?? 0).toFixed(1),
                          })}
                        </li>
                      )}
                      {!offlineMode && onlineWeather?.available && onlineWeather.current && (
                        <li>
                          {t("dash.whyOnlineLcl", {
                            nwp: onlineWeather.current.cloud_cover_pct ?? 0,
                            lcl: onlineWeather.current.estimated_cloud_base_m ?? 2000,
                          })}
                        </li>
                      )}
                    </ul>
                    {Array.isArray(selectedForecast.reasons) && selectedForecast.reasons.length > 0 && (
                      <div className="flex flex-wrap gap-1 pt-1">
                        {selectedForecast.reasons.slice(0, 4).map((r, i) => (
                          <span
                            key={i}
                            className="px-1.5 py-0.5 rounded bg-white/[0.05] border border-white/[0.08] text-[10px] font-mono text-[#9AA1AD]"
                          >
                            {tEnum(r)}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                <div className="flex items-center justify-between text-[11px] text-[#646B78] pt-1 border-t border-white/[0.06]">
                  <span className="font-mono truncate max-w-[190px]">{selectedForecast.model_version}</span>
                  <button
                    type="button"
                    onClick={() => onNavigate("forecast")}
                    className="text-[#9AA1AD] hover:text-white transition"
                  >
                    {t("overview.details")}
                  </button>
                </div>
              </>
            ) : (
              <p className="text-xs text-[#646B78] py-4 text-center">{t("overview.forecastUnavailable")}</p>
            )}
          </div>

          {/* 3. Active Alerts Block with Inline Confirm / Deny */}
          <div className="p-4 space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="obs-label">{t("dash.alerts")}</span>
              <button
                type="button"
                onClick={() => onNavigate("events")}
                className="text-[11px] text-[#9AA1AD] hover:text-white transition"
              >
                {t("overview.viewAll")}
              </button>
            </div>

            {visibleEvents.length === 0 ? (
              <p className="text-xs text-[#646B78] py-2">{t("dash.allClear")}</p>
            ) : (
              <div className="space-y-2">
                {visibleEvents.slice(0, 3).map((ev) => {
                  const status = ev.user_status ?? (ev.acknowledged ? "confirmed" : "pending");
                  const showPhoto = previewAlertObsId === ev.id && ev.observation_id != null;
                  return (
                    <div key={ev.id} className="obs-inset p-2.5 text-xs space-y-2">
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <span
                              className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                                ev.severity === "critical"
                                  ? "bg-rose-400"
                                  : ev.severity === "warning"
                                  ? "bg-amber-400"
                                  : "bg-sky-400"
                              }`}
                            />
                            <span className="font-medium text-[#F4F5F7] truncate">{ev.title}</span>
                          </div>
                          <p className="text-[11px] text-[#8E95A2] mt-0.5 line-clamp-2">{ev.description}</p>
                        </div>
                      </div>
                      <div className="flex flex-wrap items-center justify-between gap-1.5 pt-1 border-t border-white/[0.05]">
                        <div className="flex items-center gap-1.5">
                          {status === "pending" ? (
                            <>
                              <button
                                type="button"
                                disabled={alertBusyId === ev.id}
                                onClick={() => quickFeedback(ev.id, "confirmed")}
                                className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-500/15 hover:bg-emerald-500/30 border border-emerald-500/30 text-emerald-300 text-[11px] font-medium transition"
                              >
                                <Check className="w-3 h-3" /> {t("dash.confirmAlert")}
                              </button>
                              <button
                                type="button"
                                disabled={alertBusyId === ev.id}
                                onClick={() => quickFeedback(ev.id, "rejected")}
                                className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-rose-500/15 hover:bg-rose-500/30 border border-rose-500/30 text-rose-300 text-[11px] font-medium transition"
                              >
                                <X className="w-3 h-3" /> {t("dash.denyAlert")}
                              </button>
                            </>
                          ) : (
                            <span className="text-[10px] font-mono text-emerald-400">
                              ✓ {t("events.status.confirmed")}
                            </span>
                          )}
                        </div>
                        {ev.observation_id != null && (
                          <button
                            type="button"
                            onClick={() => setPreviewAlertObsId(showPhoto ? null : ev.id)}
                            className="text-[11px] font-mono text-sky-300 hover:underline"
                          >
                            {showPhoto ? t("events.hideFrame") : t("dash.viewAlertPhoto")} #{ev.observation_id}
                          </button>
                        )}
                      </div>
                      {showPhoto && ev.observation_id != null && (
                        <img
                          src={`/api/analysis/image/${ev.observation_id}?layer=segmentation`}
                          alt="Alert frame"
                          className="w-full max-h-36 object-contain rounded bg-[#08090B] border border-white/[0.08]"
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
