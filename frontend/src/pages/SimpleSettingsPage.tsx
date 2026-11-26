import React, { useEffect, useState } from "react";
import {
  Camera,
  CheckCircle2,
  Globe,
  Languages,
  LocateFixed,
  Lock,
  MapPin,
  Save,
  Search,
  Sliders,
  Sparkles,
} from "lucide-react";
import { useI18n } from "../i18n/I18nContext";
import { api } from "../services/api";

interface SimpleSettingsPageProps {
  offlineMode: boolean;
  onToggleMode: (offline: boolean) => void;
  onConfigSaved: () => void;
  onSwitchToExpert: () => void;
  onOpenSetupWizard: () => void;
}

export const SimpleSettingsPage: React.FC<SimpleSettingsPageProps> = ({
  offlineMode,
  onToggleMode,
  onConfigSaved,
  onSwitchToExpert,
  onOpenSetupWizard,
}) => {
  const { locale, setLocale, t } = useI18n();
  const [cfg, setCfg] = useState<any>(null);
  const [cityQuery, setCityQuery] = useState("");
  const [cityResults, setCityResults] = useState<any[]>([]);
  const [statusMsg, setStatusMsg] = useState("");
  const [testMsg, setTestMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.getSystemConfig().then(setCfg).catch(() => {});
  }, []);

  const flashMsg = (m: string) => {
    setStatusMsg(m);
    window.setTimeout(() => setStatusMsg(""), 3500);
  };

  const handleSaveCamera = async () => {
    if (!cfg) return;
    setBusy(true);
    try {
      await api.updateCameraConfig(cfg.camera);
      onConfigSaved();
      flashMsg(t("simple.base.saved"));
    } finally {
      setBusy(false);
    }
  };

  const handleTestCamera = async () => {
    if (!cfg) return;
    setBusy(true);
    try {
      const res = await api.testCamera({
        source_type: cfg.camera.source_type,
        source_url: cfg.camera.source_url,
        device_index: cfg.camera.device_index,
      });
      setTestMsg({
        ok: Boolean(res.ok),
        text: res.ok ? t("simple.base.camOk") : res.message || t("simple.base.camFail"),
      });
    } finally {
      setBusy(false);
    }
  };

  const handleCitySearch = async () => {
    if (!cityQuery.trim()) return;
    try {
      const res = await api.searchLocation(cityQuery.trim(), locale);
      setCityResults(res.results || []);
    } catch {
      setCityResults([]);
    }
  };

  const handleSelectCity = async (item: any) => {
    const label = [item.name, item.admin1 || item.country].filter(Boolean).join(", ");
    await api.updateRadarLocation({
      latitude: Number(item.latitude),
      longitude: Number(item.longitude),
      location_description: label,
    });
    setCityResults([]);
    setCityQuery(label);
    const fresh = await api.getSystemConfig();
    setCfg(fresh);
    onConfigSaved();
    flashMsg(t("simple.base.locationSaved", { place: label }));
  };

  const handleUseGps = () => {
    if (!navigator.geolocation) {
      flashMsg(t("radar.gpsUnsupported"));
      return;
    }
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        await api.updateRadarLocation({
          latitude: Number(pos.coords.latitude.toFixed(5)),
          longitude: Number(pos.coords.longitude.toFixed(5)),
        });
        const fresh = await api.getSystemConfig();
        setCfg(fresh);
        onConfigSaved();
        flashMsg(t("simple.base.gpsSaved"));
      },
      () => flashMsg(t("radar.gpsUnsupported"))
    );
  };

  if (!cfg) {
    return <div className="p-8 text-center text-sm text-[#8E95A2]">{t("settings.loading")}</div>;
  }

  return (
    <div className="max-w-[960px] mx-auto space-y-5 py-1">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold text-white">{t("simple.base.title")}</h2>
          <p className="text-sm text-[#9AA1AD]">{t("simple.base.subtitle")}</p>
        </div>
        {statusMsg && (
          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-emerald-500/15 border border-emerald-400/30 text-emerald-300 text-xs font-semibold">
            <CheckCircle2 className="w-4 h-4" />
            {statusMsg}
          </span>
        )}
      </div>

      {/* 1. Simple Camera Choice */}
      <div className="rounded-2xl border border-white/[0.1] bg-[#151A22] p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Camera className="w-5 h-5 text-sky-400" />
            {t("simple.base.camTitle")}
          </h3>
          <button
            type="button"
            onClick={onOpenSetupWizard}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-sky-500/15 hover:bg-sky-500/25 border border-sky-400/30 text-sky-200 text-xs font-semibold transition"
          >
            <Sparkles className="w-3.5 h-3.5" />
            {t("simple.base.wizardBtn")}
          </button>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          {[
            { id: "webcam_device", label: t("simple.base.srcWebcam") },
            { id: "http_snapshot", label: t("simple.base.srcPhone") },
            { id: "web_stream", label: t("simple.base.srcWeb") },
            { id: "demo_simulation", label: t("simple.base.srcDemo") },
          ].map((opt) => {
            const active = cfg.camera.source_type === opt.id;
            return (
              <button
                key={opt.id}
                type="button"
                onClick={() => setCfg({ ...cfg, camera: { ...cfg.camera, source_type: opt.id } })}
                className={`p-3 rounded-xl text-xs sm:text-sm font-semibold border transition text-center ${
                  active
                    ? "bg-sky-400 text-[#07131D] border-sky-300"
                    : "bg-[#10141B] text-[#CBD5E1] border-white/10 hover:border-white/25"
                }`}
              >
                {opt.label}
              </button>
            );
          })}
        </div>

        {(cfg.camera.source_type === "http_snapshot" ||
          cfg.camera.source_type === "mjpeg_stream" ||
          cfg.camera.source_type === "rtsp" ||
          cfg.camera.source_type === "web_stream") && (
          <div className="space-y-1.5">
            <label className="text-xs text-[#9AA1AD]">{t("simple.base.urlLabel")}</label>
            <input
              type="text"
              value={cfg.camera.source_url || ""}
              onChange={(e) => setCfg({ ...cfg, camera: { ...cfg.camera, source_url: e.target.value } })}
              placeholder="http://192.168.1.50:8080/shot.jpg"
              className="obs-input w-full text-sm font-mono"
            />
            {cfg.camera.source_type === "web_stream" && (
              <p className="text-xs text-amber-300/90">{t("settings.webStreamTosNotice")}</p>
            )}
          </div>
        )}

        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            disabled={busy}
            onClick={handleTestCamera}
            className="px-4 py-2 rounded-xl bg-[#10141B] hover:bg-[#1B2230] border border-white/15 text-sm font-medium text-white transition"
          >
            {t("simple.base.testBtn")}
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={handleSaveCamera}
            className="inline-flex items-center gap-1.5 px-5 py-2 rounded-xl bg-sky-400 hover:bg-sky-300 text-[#07131D] text-sm font-bold transition"
          >
            <Save className="w-4 h-4" />
            {t("simple.base.saveBtn")}
          </button>
          {testMsg && (
            <span
              className={`text-xs font-semibold ${testMsg.ok ? "text-emerald-300" : "text-rose-300"}`}
            >
              {testMsg.text}
            </span>
          )}
        </div>
      </div>

      {/* 2. Simple Location Picker */}
      <div className="rounded-2xl border border-white/[0.1] bg-[#151A22] p-5 space-y-3">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <MapPin className="w-5 h-5 text-emerald-400" />
          {t("simple.base.locTitle")}
        </h3>
        <p className="text-xs text-[#9AA1AD]">
          {t("simple.base.locCurrent")}:{" "}
          <strong className="text-white">{cfg.camera.location_description || t("simple.map.cameraPos")}</strong>
        </p>

        <div className="flex flex-wrap gap-2">
          <div className="flex-1 min-w-[200px] flex gap-2">
            <input
              type="text"
              value={cityQuery}
              onChange={(e) => setCityQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleCitySearch()}
              placeholder={t("radar.searchPlaceholder")}
              className="obs-input flex-1 text-sm"
            />
            <button
              type="button"
              onClick={handleCitySearch}
              className="px-4 py-2 rounded-xl bg-[#10141B] hover:bg-[#1B2230] border border-white/15 text-sm font-medium text-white inline-flex items-center gap-1.5"
            >
              <Search className="w-4 h-4 text-sky-300" />
              {t("radar.searchBtn")}
            </button>
          </div>
          <button
            type="button"
            onClick={handleUseGps}
            className="px-4 py-2 rounded-xl bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-400/30 text-emerald-200 text-sm font-semibold inline-flex items-center gap-1.5"
          >
            <LocateFixed className="w-4 h-4" />
            {t("radar.useGps")}
          </button>
        </div>

        {cityResults.length > 0 && (
          <div className="flex flex-wrap gap-2 pt-1">
            {cityResults.slice(0, 5).map((r, i) => (
              <button
                key={i}
                type="button"
                onClick={() => handleSelectCity(r)}
                className="px-3 py-1.5 rounded-xl bg-[#10141B] hover:bg-sky-500/20 border border-white/10 text-xs text-white transition"
              >
                {[r.name, r.admin1, r.country].filter(Boolean).join(", ")}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* 3. Online Weather & Language */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="rounded-2xl border border-white/[0.1] bg-[#151A22] p-5 space-y-3">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            {offlineMode ? <Lock className="w-5 h-5 text-[#9AA1AD]" /> : <Globe className="w-5 h-5 text-sky-400" />}
            {t("simple.base.onlineTitle")}
          </h3>
          <p className="text-xs text-[#9AA1AD]">{t("simple.base.onlineDesc")}</p>
          <button
            type="button"
            onClick={() => onToggleMode(!offlineMode)}
            className={`w-full py-2.5 rounded-xl text-sm font-bold transition ${
              offlineMode
                ? "bg-[#10141B] text-[#CBD5E1] border border-white/15 hover:border-sky-400/40"
                : "bg-emerald-400 text-[#062417]"
            }`}
          >
            {offlineMode ? t("simple.base.modeOffline") : t("simple.base.modeOnline")}
          </button>
        </div>

        <div className="rounded-2xl border border-white/[0.1] bg-[#151A22] p-5 space-y-3">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Languages className="w-5 h-5 text-sky-400" />
            {t("simple.base.langTitle")}
          </h3>
          <p className="text-xs text-[#9AA1AD]">{t("simple.base.langDesc")}</p>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => setLocale("it")}
              className={`py-2.5 rounded-xl text-sm font-bold transition ${
                locale === "it" ? "bg-sky-400 text-[#07131D]" : "bg-[#10141B] text-[#CBD5E1] border border-white/10"
              }`}
            >
              Italiano
            </button>
            <button
              type="button"
              onClick={() => setLocale("en")}
              className={`py-2.5 rounded-xl text-sm font-bold transition ${
                locale === "en" ? "bg-sky-400 text-[#07131D]" : "bg-[#10141B] text-[#CBD5E1] border border-white/10"
              }`}
            >
              English
            </button>
          </div>
        </div>
      </div>

      {/* 4. Gateway to Expert Mode */}
      <div className="rounded-2xl border border-white/[0.08] bg-[#10141B] p-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5 text-sm text-[#9AA1AD]">
          <Sliders className="w-4 h-4 text-sky-400 shrink-0" />
          <span>{t("simple.base.expertHint")}</span>
        </div>
        <button
          type="button"
          onClick={onSwitchToExpert}
          className="px-4 py-2 rounded-xl bg-[#2563EB] hover:bg-[#1D4ED8] text-white text-xs sm:text-sm font-semibold transition"
        >
          {t("simple.base.switchExpertBtn")}
        </button>
      </div>
    </div>
  );
};
