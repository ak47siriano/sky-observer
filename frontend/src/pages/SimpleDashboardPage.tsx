import React, { useState } from "react";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  Cloud,
  MapPin,
  Navigation,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Upload,
} from "lucide-react";
import { useI18n } from "../i18n/I18nContext";
import type { CameraStatus, EventAlert, ForecastItem, Observation } from "../types/api";

interface SimpleDashboardPageProps {
  camera: CameraStatus | null;
  obs: Observation | null;
  forecasts: ForecastItem[];
  events: EventAlert[];
  onlineWeather?: any;
  offlineMode: boolean;
  onTriggerSnapshot: () => void;
  onUploadFile: (file: File) => void;
  onOpenSimpleMap: () => void;
  onOpenSimpleBase: () => void;
  busy: boolean;
}

type WeatherGlyphKind = "sun" | "partly" | "cloudy" | "overcast" | "storm" | "night_clear" | "night_cloudy";

const WeatherIllustration: React.FC<{ kind: WeatherGlyphKind; size?: "lg" | "sm" }> = ({
  kind,
  size = "lg",
}) => {
  const cls = size === "lg" ? "w-24 h-24 shrink-0" : "w-12 h-12 shrink-0";

  if (kind === "sun") {
    return (
      <svg viewBox="0 0 96 96" className={cls} aria-hidden="true">
        <defs>
          <radialGradient id="sunCore" cx="45%" cy="40%" r="60%">
            <stop offset="0%" stopColor="#FEF08A" />
            <stop offset="55%" stopColor="#FBBF24" />
            <stop offset="100%" stopColor="#F59E0B" />
          </radialGradient>
        </defs>
        <g stroke="#FBBF24" strokeWidth="5" strokeLinecap="round">
          <line x1="48" y1="8" x2="48" y2="18" />
          <line x1="48" y1="78" x2="48" y2="88" />
          <line x1="8" y1="48" x2="18" y2="48" />
          <line x1="78" y1="48" x2="88" y2="48" />
          <line x1="19.7" y1="19.7" x2="26.8" y2="26.8" />
          <line x1="69.2" y1="69.2" x2="76.3" y2="76.3" />
          <line x1="76.3" y1="19.7" x2="69.2" y2="26.8" />
          <line x1="26.8" y1="69.2" x2="19.7" y2="76.3" />
        </g>
        <circle cx="48" cy="48" r="22" fill="url(#sunCore)" />
      </svg>
    );
  }

  if (kind === "partly") {
    return (
      <svg viewBox="0 0 96 96" className={cls} aria-hidden="true">
        <defs>
          <radialGradient id="sunPart" cx="45%" cy="40%" r="60%">
            <stop offset="0%" stopColor="#FEF08A" />
            <stop offset="65%" stopColor="#FBBF24" />
            <stop offset="100%" stopColor="#F59E0B" />
          </radialGradient>
          <linearGradient id="cloudWhite" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#FFFFFF" />
            <stop offset="100%" stopColor="#CBD5E1" />
          </linearGradient>
        </defs>
        <g stroke="#FBBF24" strokeWidth="4" strokeLinecap="round">
          <line x1="36" y1="10" x2="36" y2="17" />
          <line x1="10" y1="36" x2="17" y2="36" />
          <line x1="17" y1="17" x2="22" y2="22" />
          <line x1="55" y1="17" x2="50" y2="22" />
        </g>
        <circle cx="36" cy="36" r="16" fill="url(#sunPart)" />
        <path
          d="M32 72h38a14 14 0 0 0 2-27.8A18 18 0 0 0 38 40a14 14 0 0 0-6 32Z"
          fill="url(#cloudWhite)"
        />
      </svg>
    );
  }

  if (kind === "cloudy") {
    return (
      <svg viewBox="0 0 96 96" className={cls} aria-hidden="true">
        <defs>
          <linearGradient id="cloudBack" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#CBD5E1" />
            <stop offset="100%" stopColor="#94A3B8" />
          </linearGradient>
          <linearGradient id="cloudFront" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#F8FAFC" />
            <stop offset="100%" stopColor="#CBD5E1" />
          </linearGradient>
        </defs>
        <path d="M24 54h38a11 11 0 0 0 1.5-21.8A15 15 0 0 0 35 28a12 12 0 0 0-11 26Z" fill="url(#cloudBack)" />
        <path d="M26 74h44a14 14 0 0 0 2-27.8A19 19 0 0 0 36 42a15 15 0 0 0-10 32Z" fill="url(#cloudFront)" />
      </svg>
    );
  }

  if (kind === "overcast") {
    return (
      <svg viewBox="0 0 96 96" className={cls} aria-hidden="true">
        <defs>
          <linearGradient id="ovcGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#94A3B8" />
            <stop offset="100%" stopColor="#475569" />
          </linearGradient>
        </defs>
        <path d="M20 70h52a15 15 0 0 0 2-29.8A21 21 0 0 0 34 35a16 16 0 0 0-14 35Z" fill="url(#ovcGrad)" />
      </svg>
    );
  }

  if (kind === "storm") {
    return (
      <svg viewBox="0 0 96 96" className={cls} aria-hidden="true">
        <defs>
          <linearGradient id="stormGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#64748B" />
            <stop offset="100%" stopColor="#334155" />
          </linearGradient>
        </defs>
        <path d="M20 62h52a14 14 0 0 0 2-27.8A20 20 0 0 0 34 30a15 15 0 0 0-14 32Z" fill="url(#stormGrad)" />
        <polygon points="48,58 38,76 47,76 44,90 58,70 49,70" fill="#FBBF24" />
        <line x1="30" y1="68" x2="26" y2="79" stroke="#38BDF8" strokeWidth="3.5" strokeLinecap="round" />
        <line x1="66" y1="68" x2="62" y2="79" stroke="#38BDF8" strokeWidth="3.5" strokeLinecap="round" />
      </svg>
    );
  }

  if (kind === "night_clear") {
    return (
      <svg viewBox="0 0 96 96" className={cls} aria-hidden="true">
        <path
          d="M58 20a26 26 0 1 0 20 42 21 21 0 1 1-20-42Z"
          fill="#E2E8F0"
        />
        <circle cx="68" cy="24" r="3" fill="#FDE68A" />
        <circle cx="78" cy="38" r="2.2" fill="#BAE6FD" />
        <circle cx="24" cy="26" r="2" fill="#E0F2FE" />
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 96 96" className={cls} aria-hidden="true">
      <path d="M52 16a18 18 0 1 0 14 29 14 14 0 1 1-14-29Z" fill="#CBD5E1" />
      <path d="M22 72h48a14 14 0 0 0 2-27.8A18 18 0 0 0 36 40a14 14 0 0 0-14 32Z" fill="#64748B" />
    </svg>
  );
};

function directionFullName(cardinal: string, locale: "en" | "it"): string {
  const itMap: Record<string, string> = {
    N: "Nord",
    NE: "Nord-Est",
    E: "Est",
    SE: "Sud-Est",
    S: "Sud",
    SW: "Sud-Ovest",
    W: "Ovest",
    NW: "Nord-Ovest",
  };
  const enMap: Record<string, string> = {
    N: "North",
    NE: "Northeast",
    E: "East",
    SE: "Southeast",
    S: "South",
    SW: "Southwest",
    W: "West",
    NW: "Northwest",
  };
  const key = (cardinal || "N").toUpperCase();
  return (locale === "it" ? itMap[key] : enMap[key]) || key;
}

export const SimpleDashboardPage: React.FC<SimpleDashboardPageProps> = ({
  camera,
  obs,
  forecasts,
  events,
  onlineWeather,
  offlineMode,
  onTriggerSnapshot,
  onUploadFile,
  onOpenSimpleMap,
  onOpenSimpleBase,
  busy,
}) => {
  const { locale, t } = useI18n();
  const [photoLayer, setPhotoLayer] = useState<"raw" | "segmentation">("raw");

  const cov = obs?.cloud_coverage_pct ?? 0;
  const dark = obs?.dark_cloud_pct ?? 0;
  const skyRegime = obs?.coverage?.sky_regime || "day";
  const isNight = skyRegime === "night";

  // Derive big human-friendly sky verdict
  let heroKind: WeatherGlyphKind = "partly";
  let heroTitle = t("simple.verdict.partly");
  let statusBadge = t("simple.status.calm");
  let statusTone: "emerald" | "sky" | "amber" | "rose" = "emerald";

  if (!obs) {
    heroKind = "partly";
    heroTitle = t("simple.verdict.waiting");
    statusBadge = t("simple.status.setup");
    statusTone = "sky";
  } else if (dark >= 32 || obs.visual_condition === "dark_storm_like_clouds") {
    heroKind = "storm";
    heroTitle = t("simple.verdict.storm");
    statusBadge = t("simple.status.storm");
    statusTone = "rose";
  } else if (isNight) {
    heroKind = cov < 30 ? "night_clear" : "night_cloudy";
    heroTitle = cov < 30 ? t("simple.verdict.nightClear") : t("simple.verdict.nightCloudy");
    statusBadge = t("simple.status.calm");
    statusTone = "sky";
  } else if (cov < 15) {
    heroKind = "sun";
    heroTitle = t("simple.verdict.clear");
    statusBadge = t("simple.status.calm");
    statusTone = "emerald";
  } else if (cov < 45) {
    heroKind = "partly";
    heroTitle = t("simple.verdict.partly");
    statusBadge = t("simple.status.calm");
    statusTone = "emerald";
  } else if (cov < 78) {
    heroKind = "cloudy";
    heroTitle = t("simple.verdict.cloudy");
    statusBadge = t("simple.status.variable");
    statusTone = "sky";
  } else {
    heroKind = "overcast";
    heroTitle = dark >= 20 ? t("simple.verdict.darkOvercast") : t("simple.verdict.overcast");
    statusBadge = dark >= 20 ? t("simple.status.watch") : t("simple.status.variable");
    statusTone = dark >= 20 ? "amber" : "sky";
  }

  // Plain-language coverage sentence
  const coverSummary = !obs
    ? t("simple.cover.none")
    : cov < 15
    ? t("simple.cover.veryClear", { pct: Math.round(cov) })
    : cov < 40
    ? t("simple.cover.mostlyClear", { pct: Math.round(cov) })
    : cov < 70
    ? t("simple.cover.mixed", { pct: Math.round(cov) })
    : t("simple.cover.mostlyCloudy", { pct: Math.round(cov) });

  // Pick up to 3 simple forecast cards (+5m, +15m, +20/+30m)
  const simpleForecastCards = (() => {
    const valid = forecasts.filter((f) => f.status === "available");
    if (!valid.length) return [];
    const pick = [
      valid.find((f) => f.horizon_minutes === 5) || valid[0],
      valid.find((f) => f.horizon_minutes === 15) || valid[1],
      valid.find((f) => f.horizon_minutes === 30) ||
        valid.find((f) => f.horizon_minutes === 20) ||
        valid[valid.length - 1],
    ].filter((x, i, arr): x is ForecastItem => Boolean(x) && arr.findIndex((y) => y?.id === x?.id) === i);

    return pick.slice(0, 3).map((fc) => {
      const pred = fc.cloud_cover_prediction;
      const diff = obs ? pred - obs.cloud_coverage_pct : 0;
      let kind: WeatherGlyphKind = "partly";
      let label = t("simple.fc.steady");

      if (fc.storm_signal_prob >= 0.45 || fc.darkening_prob >= 0.6) {
        kind = "storm";
        label = t("simple.fc.darkening");
      } else if (pred < 18) {
        kind = isNight ? "night_clear" : "sun";
        label = diff <= -10 ? t("simple.fc.clearing") : t("simple.fc.sunny");
      } else if (diff <= -10 || fc.clearing_prob >= 0.55) {
        kind = isNight ? "night_clear" : "partly";
        label = t("simple.fc.clearing");
      } else if (diff >= 10 || fc.increasing_prob >= 0.55) {
        kind = pred >= 75 ? "overcast" : "cloudy";
        label = t("simple.fc.moreClouds");
      } else if (pred >= 80) {
        kind = "overcast";
        label = t("simple.fc.cloudy");
      } else if (pred >= 45) {
        kind = "cloudy";
        label = t("simple.fc.steady");
      } else {
        kind = isNight ? "night_cloudy" : "partly";
        label = t("simple.fc.steady");
      }

      return {
        id: fc.id,
        horizon: fc.horizon_minutes,
        kind,
        label,
      };
    });
  })();

  // Practical traffic-light advice
  const hasCriticalAlert = events.some(
    (e) =>
      (e.user_status ?? "pending") !== "rejected" &&
      (e.severity === "critical" || e.event_type === "storm_like_development")
  );
  const maxPrecipProb = forecasts.reduce((acc, f) => Math.max(acc, f.precip_signal_prob || 0), 0);
  const maxIncreaseProb = forecasts.reduce((acc, f) => Math.max(acc, f.increasing_prob || 0), 0);

  let adviceLevel: "green" | "amber" | "rose" = "green";
  let adviceTitle = t("simple.advice.greenTitle");
  let adviceBody = t("simple.advice.greenBody");
  if (hasCriticalAlert || dark >= 35 || maxPrecipProb >= 0.55) {
    adviceLevel = "rose";
    adviceTitle = t("simple.advice.roseTitle");
    adviceBody = t("simple.advice.roseBody");
  } else if (dark >= 18 || cov >= 75 || maxIncreaseProb >= 0.6) {
    adviceLevel = "amber";
    adviceTitle = t("simple.advice.amberTitle");
    adviceBody = t("simple.advice.amberBody");
  }

  const moving = Boolean(obs?.motion?.available && obs.motion_speed_px_s >= 0.15);
  const dirLabel = moving
    ? t("simple.map.movingToward", { dir: directionFullName(obs!.motion_direction_cardinal, locale) })
    : t("simple.map.calmClouds");
  const locationLabel =
    onlineWeather?.location_name || camera?.location_description || t("simple.map.cameraPos");
  const tempC =
    !offlineMode && onlineWeather?.available && onlineWeather?.current?.temperature_c != null
      ? `${Math.round(Number(onlineWeather.current.temperature_c))}°C`
      : null;

  const badgeToneCls = {
    emerald: "bg-emerald-400/90 text-[#062417]",
    sky: "bg-sky-400/90 text-[#061E2E]",
    amber: "bg-amber-400/90 text-[#291902]",
    rose: "bg-rose-500/90 text-white",
  }[statusTone];

  const adviceToneCls = {
    green: "border-emerald-400/40 bg-emerald-950/30 text-emerald-100",
    amber: "border-amber-400/40 bg-amber-950/30 text-amber-100",
    rose: "border-rose-500/45 bg-rose-950/35 text-rose-100",
  }[adviceLevel];

  return (
    <div className="max-w-[1440px] mx-auto space-y-5 py-1">
      {!obs && (
        <div className="obs-panel p-6 rounded-2xl border-sky-500/30 bg-[#151A22] flex flex-wrap items-center justify-between gap-4">
          <div className="space-y-1">
            <h2 className="text-lg font-bold text-white">{t("simple.empty.title")}</h2>
            <p className="text-sm text-[#9AA1AD]">{t("simple.empty.subtitle")}</p>
          </div>
          <div className="flex flex-wrap items-center gap-2.5">
            <label className="obs-btn-secondary cursor-pointer inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium">
              <Upload className="w-4 h-4 text-sky-300" />
              {t("simple.empty.upload")}
              <input
                type="file"
                accept="image/*,video/*"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && onUploadFile(e.target.files[0])}
              />
            </label>
            <button
              type="button"
              onClick={onOpenSimpleBase}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-sky-500 hover:bg-sky-400 text-[#07131D] text-sm font-bold transition"
            >
              <Camera className="w-4 h-4" />
              {t("simple.empty.setupCam")}
            </button>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* LEFT COLUMN: Clean Live Sky Window + Simple Visual Cloud Direction Card */}
        <div className="lg:col-span-7 space-y-5">
          {/* Live Sky Photo Card */}
          <div className="relative rounded-2xl overflow-hidden border border-white/[0.1] bg-[#0B0E14] shadow-xl aspect-[16/10]">
            {obs ? (
              <img
                src={`/api/camera/latest?layer=${photoLayer}&t=${obs.id}`}
                alt={t("simple.photo.live")}
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full flex flex-col items-center justify-center text-[#7C8596] gap-3 p-6 text-center">
                <Cloud className="w-14 h-14 text-sky-400/50" />
                <p className="text-sm">{t("simple.photo.waiting")}</p>
              </div>
            )}

            {/* Simple 2-Pill Overlay Controls */}
            <div className="absolute top-3.5 left-3.5 right-3.5 flex flex-wrap items-center justify-between gap-2">
              <div className="inline-flex items-center p-1 rounded-full bg-[#0D1117]/80 backdrop-blur-md border border-white/15 shadow-lg">
                <button
                  type="button"
                  onClick={() => setPhotoLayer("raw")}
                  className={`px-4 py-1.5 rounded-full text-xs sm:text-sm font-semibold transition ${
                    photoLayer === "raw"
                      ? "bg-sky-400 text-[#07131D] shadow"
                      : "text-[#D4DBE6] hover:text-white"
                  }`}
                >
                  {t("simple.photo.live")}
                </button>
                <button
                  type="button"
                  onClick={() => setPhotoLayer("segmentation")}
                  className={`px-4 py-1.5 rounded-full text-xs sm:text-sm font-semibold transition ${
                    photoLayer === "segmentation"
                      ? "bg-sky-400 text-[#07131D] shadow"
                      : "text-[#D4DBE6] hover:text-white"
                  }`}
                >
                  {t("simple.photo.highlightClouds")}
                </button>
              </div>

              <button
                type="button"
                disabled={busy}
                onClick={onTriggerSnapshot}
                className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-[#0D1117]/80 hover:bg-[#161D28]/90 backdrop-blur-md border border-white/15 text-xs sm:text-sm font-medium text-white transition"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${busy ? "animate-spin" : ""}`} />
                <span className="hidden sm:inline">{t("simple.photo.refresh")}</span>
              </button>
            </div>

            {obs && (
              <div className="absolute bottom-3 left-3.5 px-3 py-1 rounded-full bg-[#0D1117]/75 backdrop-blur-md border border-white/10 text-xs text-[#E2E8F0]">
                {locationLabel}
              </div>
            )}
          </div>

          {/* Simplified "Where are the clouds?" Visual Direction Card (NO house — clean camera pin!) */}
          <button
            type="button"
            onClick={onOpenSimpleMap}
            className="w-full text-left group rounded-2xl border border-white/[0.09] bg-gradient-to-br from-[#161C26] via-[#131820] to-[#0E131A] hover:border-sky-400/40 p-5 transition shadow-lg relative overflow-hidden"
          >
            <div className="flex items-center justify-between gap-2 mb-3">
              <h3 className="text-base sm:text-lg font-bold text-white">{t("simple.map.cardTitle")}</h3>
              <span className="text-xs font-semibold text-sky-300 group-hover:translate-x-0.5 transition">
                {t("simple.map.openFull")} →
              </span>
            </div>

            <div className="relative h-36 rounded-xl bg-[#0D1219] border border-white/[0.06] overflow-hidden flex items-center justify-around px-6">
              {/* Subtle stylized background cloud puffs based on real cloud cover */}
              {cov >= 12 && (
                <div className="absolute left-6 bottom-3 w-24 h-12 rounded-full bg-white/15 blur-md pointer-events-none" />
              )}
              {cov >= 35 && (
                <div className="absolute right-8 top-3 w-28 h-14 rounded-full bg-white/20 blur-md pointer-events-none" />
              )}

              {/* Camera Location Pin (never "Casa tua") */}
              <div className="relative z-10 flex flex-col items-center gap-1.5">
                <div className="w-12 h-12 rounded-full bg-sky-400/20 border-2 border-sky-400 flex items-center justify-center shadow-[0_0_24px_rgba(56,189,248,0.35)]">
                  <MapPin className="w-6 h-6 text-sky-300" />
                </div>
                <span className="text-xs sm:text-sm font-semibold text-white max-w-[170px] truncate">
                  {locationLabel}
                </span>
              </div>

              {/* Big Intuitive Cloud Drift Arrow */}
              <div className="relative z-10 flex items-center gap-4">
                <div className="w-14 h-14 rounded-2xl bg-sky-500/15 border border-sky-400/35 flex items-center justify-center">
                  <Navigation
                    className="w-8 h-8 text-sky-400 transition-transform duration-500"
                    style={{
                      transform: moving ? `rotate(${obs!.motion_direction_deg}deg)` : "rotate(0deg)",
                      opacity: moving ? 1 : 0.45,
                    }}
                  />
                </div>
                <div className="max-w-[200px]">
                  <div className="text-sm sm:text-base font-bold text-white leading-snug">{dirLabel}</div>
                  <div className="text-xs text-[#9AA1AD] mt-0.5">
                    {obs ? `${Math.round(cov)}% ${t("simple.map.cloudsWord")}` : t("simple.cover.none")}
                  </div>
                </div>
              </div>
            </div>
          </button>
        </div>

        {/* RIGHT COLUMN: Hero Verdict + 3-Icon Plain Forecast + Traffic-Light Advice */}
        <div className="lg:col-span-5 space-y-5">
          {/* 1. Big Hero Sky Verdict Card */}
          <div className="rounded-2xl border border-white/[0.1] bg-[#151A22] p-6 space-y-5 shadow-xl">
            <div className="flex items-center gap-5">
              <WeatherIllustration kind={heroKind} size="lg" />
              <div className="min-w-0 flex-1 space-y-2">
                <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white leading-tight">
                  {heroTitle}
                </h1>
                <div className="flex flex-wrap items-center gap-3">
                  <span className={`px-3.5 py-1 rounded-full text-xs sm:text-sm font-bold ${badgeToneCls}`}>
                    {statusBadge}
                  </span>
                  {tempC && (
                    <span className="text-3xl font-bold text-white tabular-nums tracking-tight">{tempC}</span>
                  )}
                </div>
              </div>
            </div>

            {/* Clean Visual Blue Sky vs Cloud Bar */}
            <div className="space-y-2">
              <div className="h-3.5 rounded-full overflow-hidden flex bg-white/10 p-0.5">
                <div
                  style={{ width: `${Math.max(4, Math.min(100, 100 - cov))}%` }}
                  className="bg-gradient-to-r from-sky-500 to-sky-300 rounded-l-full transition-all duration-700"
                />
                <div
                  style={{ width: `${Math.max(0, Math.min(100, cov))}%` }}
                  className="bg-slate-200 rounded-r-full transition-all duration-700"
                />
              </div>
              <div className="text-center text-sm text-[#CBD5E1] font-medium">{coverSummary}</div>
            </div>
          </div>

          {/* 2. Simple 3-Step Visual Forecast ("Tra poco cosa succede?") */}
          <div className="rounded-2xl border border-white/[0.1] bg-[#151A22] p-5 space-y-4 shadow-xl">
            <h2 className="text-base sm:text-lg font-bold text-white">{t("simple.fc.title")}</h2>
            {simpleForecastCards.length > 0 ? (
              <div className="grid grid-cols-3 gap-3">
                {simpleForecastCards.map((item) => (
                  <div
                    key={item.id}
                    className="rounded-xl bg-[#10141B] border border-white/[0.07] p-3.5 flex flex-col items-center text-center gap-2"
                  >
                    <span className="text-xs sm:text-sm font-medium text-[#9AA1AD]">
                      {t("simple.fc.inMin", { min: item.horizon })}
                    </span>
                    <WeatherIllustration kind={item.kind} size="sm" />
                    <span className="text-sm sm:text-base font-bold text-white leading-tight">{item.label}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded-xl bg-[#10141B] border border-white/[0.06] p-5 text-center text-sm text-[#8E95A2]">
                {t("simple.fc.waiting")}
              </div>
            )}
          </div>

          {/* 3. Practical Advice Traffic-Light Banner */}
          <div className={`rounded-2xl border p-5 flex items-start gap-4 shadow-lg ${adviceToneCls}`}>
            <div className="shrink-0 mt-0.5">
              {adviceLevel === "green" ? (
                <div className="w-11 h-11 rounded-xl bg-emerald-400/20 border border-emerald-400/40 flex items-center justify-center">
                  <ShieldCheck className="w-6 h-6 text-emerald-300" />
                </div>
              ) : adviceLevel === "amber" ? (
                <div className="w-11 h-11 rounded-xl bg-amber-400/20 border border-amber-400/40 flex items-center justify-center">
                  <AlertTriangle className="w-6 h-6 text-amber-300" />
                </div>
              ) : (
                <div className="w-11 h-11 rounded-xl bg-rose-500/25 border border-rose-400/50 flex items-center justify-center">
                  <ShieldAlert className="w-6 h-6 text-rose-300" />
                </div>
              )}
            </div>
            <div className="space-y-1">
              <div className="text-base sm:text-lg font-bold text-white leading-snug">{adviceTitle}</div>
              <p className="text-xs sm:text-sm opacity-90 leading-relaxed">{adviceBody}</p>
            </div>
          </div>

          {/* Tiny Friendly Offline/Online Hint */}
          {offlineMode && (
            <div className="flex items-center gap-2 text-xs text-[#8E95A2] px-1">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
              <span>{t("simple.offlineNote")}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
