import React, { useCallback, useEffect, useState } from "react";
import {
  Bell,
  CloudSun,
  Cpu,
  Globe,
  HelpCircle,
  Languages,
  LayoutDashboard,
  Lock,
  MapPin,
  Settings,
} from "lucide-react";
import { ManualModal } from "./components/ManualModal";
import { useI18n } from "./i18n/I18nContext";
import { api } from "./services/api";
import type { EventAlert, ForecastItem, Observation, SystemStatus } from "./types/api";
import { AiStudioPage } from "./pages/AiStudioPage";
import { AnalyticsPage } from "./pages/AnalyticsPage";
import { DashboardPage } from "./pages/DashboardPage";
import { EventsPage } from "./pages/EventsPage";
import { RadarMapPage } from "./pages/RadarMapPage";
import { SettingsPage } from "./pages/SettingsPage";

const NAV_ITEMS = [
  { id: "dashboard", labelKey: "nav.dashboard", icon: LayoutDashboard },
  { id: "radar", labelKey: "nav.radar", icon: MapPin },
  { id: "forecast", labelKey: "nav.forecast", icon: CloudSun },
  { id: "events", labelKey: "nav.events", icon: Bell },
  { id: "ai", labelKey: "nav.ai", icon: Cpu },
  { id: "settings", labelKey: "nav.settings", icon: Settings },
] as const;

export function App() {
  const { locale, setLocale, t } = useI18n();
  const [page, setPage] = useState<string>(() => {
    const q = new URLSearchParams(window.location.search).get("page");
    return q && ["dashboard", "radar", "forecast", "events", "ai", "settings"].includes(q) ? q : "dashboard";
  });
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [latestObs, setLatestObs] = useState<Observation | null>(null);
  const [forecasts, setForecasts] = useState<ForecastItem[]>([]);
  const [events, setEvents] = useState<EventAlert[]>([]);
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState<string>("");
  const [manualOpen, setManualOpen] = useState(false);
  const [utcTime, setUtcTime] = useState<string>(() => new Date().toISOString().slice(11, 19));

  useEffect(() => {
    const clockId = window.setInterval(() => {
      setUtcTime(new Date().toISOString().slice(11, 19));
    }, 1000);
    return () => window.clearInterval(clockId);
  }, []);

  const refreshAll = useCallback(async () => {
    try {
      const [sys, obsRes, fcRes, evRes] = await Promise.all([
        api.getSystemStatus(),
        api.getLatestAnalysis(),
        api.getLatestForecast(),
        api.getEvents(50),
      ]);
      setSystem(sys);
      setLatestObs(obsRes.available ? obsRes.observation : null);
      setForecasts(fcRes.horizons || []);
      setEvents(evRes.items || []);
    } catch {
      // ignore transient startup poll
    }
  }, []);

  useEffect(() => {
    refreshAll();
    let wsConnected = false;
    let lastSecondarySync = 0;
    let ws: WebSocket | null = null;
    let reconnectTimer: number | null = null;
    let disposed = false;

    const refreshSecondary = async () => {
      const now = Date.now();
      if (now - lastSecondarySync < 4000) return;
      lastSecondarySync = now;
      try {
        const [sys, fcRes, evRes] = await Promise.all([
          api.getSystemStatus(),
          api.getLatestForecast(),
          api.getEvents(50),
        ]);
        setSystem(sys);
        setForecasts(fcRes.horizons || []);
        setEvents(evRes.items || []);
      } catch {
        // ignore transient errors
      }
    };

    const connectWs = () => {
      if (disposed) return;
      const proto = window.location.protocol === "https:" ? "wss" : "ws";
      try {
        ws = new WebSocket(`${proto}://${window.location.host}/ws/live`);
        ws.onopen = () => {
          wsConnected = true;
        };
        ws.onmessage = (evt) => {
          try {
            const msg = JSON.parse(evt.data);
            if (msg.type === "observation" && msg.payload) {
              setLatestObs(msg.payload);
              void refreshSecondary();
            }
          } catch {
            // ignore
          }
        };
        ws.onclose = () => {
          wsConnected = false;
          if (!disposed) {
            reconnectTimer = window.setTimeout(connectWs, 4000);
          }
        };
        ws.onerror = () => {
          wsConnected = false;
        };
      } catch {
        wsConnected = false;
      }
    };

    connectWs();
    // Fallback poll only when WebSocket is disconnected
    const fallbackTimer = window.setInterval(() => {
      if (!wsConnected) {
        void refreshAll();
      }
    }, 10000);

    return () => {
      disposed = true;
      window.clearInterval(fallbackTimer);
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, [refreshAll]);

  const showToast = (m: string) => {
    setToast(m);
    window.setTimeout(() => setToast(""), 3500);
  };

  const handleToggleOfflineMode = async (offline: boolean) => {
    setBusy(true);
    try {
      await api.setSystemMode(offline);
      await refreshAll();
      showToast(offline ? t("mode.switchedOffline") : t("mode.switchedOnline"));
    } catch (e: any) {
      showToast(e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleTriggerSnapshot = async () => {
    setBusy(true);
    try {
      await api.triggerSnapshot();
      await refreshAll();
      showToast(t("app.capturedToast"));
    } catch (e: any) {
      showToast(t("app.captureErrToast", { msg: e.message }));
    } finally {
      setBusy(false);
    }
  };

  const handleUploadFile = async (file: File) => {
    setBusy(true);
    try {
      const res = await api.uploadCameraFile(file);
      await refreshAll();
      showToast(t("app.importedToast", { n: res.imported_frames }));
    } catch (e: any) {
      showToast(t("app.uploadErrToast", { msg: e.message }));
    } finally {
      setBusy(false);
    }
  };

  const unackCount = events.filter(
    (e) => (e.user_status ?? (e.acknowledged ? "confirmed" : "pending")) === "pending"
  ).length;
  const offlineMode = system?.offline_mode ?? true;

  return (
    <div className="min-h-screen text-[#ECEFF4] flex flex-col pb-16 sm:pb-0">
      {/* Top Meteorological Station Bar */}
      <header className="sticky top-0 z-40 border-b border-[#262B34] bg-[#15181E]">
        <div className="max-w-[1800px] w-full mx-auto px-3 sm:px-4 h-13 py-2 flex items-center justify-between gap-4">
          {/* Brand */}
          <button
            type="button"
            onClick={() => setPage("dashboard")}
            className="flex items-center gap-2.5 shrink-0 text-left"
          >
            <div className="relative">
              <img
                src="/logo.png"
                alt="Sky Observer"
                className="w-8 h-8 rounded object-cover border border-[#343B48] bg-[#101216]"
              />
              <span
                className={`absolute -bottom-0.5 -right-0.5 w-2 h-2 rounded-full ring-2 ring-[#15181E] ${
                  system?.camera.connected ? "bg-emerald-500" : "bg-rose-500"
                }`}
              />
            </div>
            <div className="hidden sm:block leading-tight">
              <div className="text-[13px] font-bold tracking-wider text-[#ECEFF4] flex items-center gap-1.5">
                <span>SKY OBSERVER</span>
                {latestObs && (
                  <span className="px-1.5 py-0.5 rounded bg-[#1B1F27] border border-[#343B48] text-[#ECEFF4] text-[10px] font-mono tabular-nums">
                    {latestObs.cloud_coverage_pct < 6
                      ? "SKC"
                      : latestObs.cloud_coverage_pct < 31
                      ? "FEW"
                      : latestObs.cloud_coverage_pct < 56
                      ? "SCT"
                      : latestObs.cloud_coverage_pct < 88
                      ? "BKN"
                      : "OVC"}{" "}
                    {Math.round((latestObs.cloud_coverage_pct / 100) * 8)}/8
                  </span>
                )}
              </div>
              <div className="text-[10px] font-mono text-[#8D96A5] tracking-wide">
                {offlineMode ? "LOCAL · OFFLINE" : "HYBRID · OPEN-METEO"}
              </div>
            </div>
          </button>

          {/* 6-Item Streamlined Desktop Navigation */}
          <nav className="hidden sm:flex items-center gap-1 p-0.5 rounded bg-[#101216] border border-[#262B34] overflow-x-auto no-scrollbar">
            {NAV_ITEMS.map(({ id, labelKey, icon: Icon }) => {
              const active = page === id;
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => setPage(id)}
                  className={active ? "obs-tab-active" : "obs-tab"}
                >
                  <Icon className={`w-3.5 h-3.5 ${active ? "text-sky-300" : ""}`} />
                  <span>{t(labelKey)}</span>
                  {id === "events" && unackCount > 0 && (
                    <span className="ml-0.5 px-1.5 py-0.2 rounded-full bg-amber-500/25 border border-amber-400/30 text-amber-200 text-[10px] font-mono tabular-nums">
                      {unackCount}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>

          {/* System Mode, Locale & Hardware Pill */}
          <div className="flex items-center gap-2 shrink-0 text-[11px]">
            <button
              type="button"
              disabled={busy}
              onClick={() => handleToggleOfflineMode(!offlineMode)}
              title="Toggle Offline Only vs Online Hybrid (Open-Meteo + RainViewer)"
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded border border-[#343B48] bg-[#1B1F27] hover:bg-[#222731] text-[#ECEFF4] font-medium transition-colors"
            >
              {offlineMode ? <Lock className="w-3 h-3 text-[#8D96A5]" /> : <Globe className="w-3 h-3 text-[#8D96A5]" />}
              <span className="hidden md:inline">
                {offlineMode ? t("mode.offlineOnly") : t("mode.onlineHybrid")}
              </span>
            </button>

            <div className="inline-flex items-center bg-[#101216] border border-[#262B34] rounded p-0.5">
              <Languages className="w-3 h-3 text-[#5D6675] mx-1 hidden sm:block" />
              <button
                type="button"
                onClick={() => setLocale("en")}
                className={`px-2 py-0.5 rounded text-[10px] font-semibold transition-colors ${
                  locale === "en" ? "bg-[#2563EB] text-white" : "text-[#8D96A5] hover:text-white"
                }`}
              >
                EN
              </button>
              <button
                type="button"
                onClick={() => setLocale("it")}
                className={`px-2 py-0.5 rounded text-[10px] font-semibold transition-colors ${
                  locale === "it" ? "bg-[#2563EB] text-white" : "text-[#8D96A5] hover:text-white"
                }`}
              >
                IT
              </button>
            </div>

            <div
              title={
                system
                  ? `${system.hardware.gpu.active_device.toUpperCase()} · RAM ${system.hardware.ram_percent}% · Storage ${system.storage.total_app_storage_mb} MB`
                  : ""
              }
              className="hidden lg:inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded bg-[#101216] border border-[#262B34] font-mono tabular-nums text-[#8D96A5]"
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  system?.camera.connected ? "bg-emerald-500" : "bg-rose-500"
                }`}
              />
              <span className="text-[#ECEFF4]">{system?.camera.connected ? "LIVE" : "OFF"}</span>
              <span className="text-[#ECEFF4]">· {utcTime}Z</span>
              {system && (
                <span className="text-[#5D6675]">
                  · {system.hardware.gpu.active_device.toUpperCase()} {system.hardware.ram_percent}%
                </span>
              )}
            </div>

            {/* Context-Aware Manual (?) Button */}
            <button
              type="button"
              onClick={() => setManualOpen(true)}
              title={locale === "it" ? "Apri il Manuale (sezione pagina corrente)" : "Open Manual (current page section)"}
              aria-label={locale === "it" ? "Apri il Manuale" : "Open Manual"}
              className="w-8 h-8 rounded bg-[#1B1F27] hover:bg-[#222731] border border-[#343B48] text-[#8D96A5] hover:text-[#ECEFF4] inline-flex items-center justify-center transition-colors"
            >
              <HelpCircle className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 w-full max-w-[1800px] mx-auto px-3 sm:px-4 py-3">
        {page === "dashboard" && (
          <DashboardPage
            camera={system?.camera || null}
            obs={latestObs}
            forecasts={forecasts}
            events={events}
            onlineWeather={system?.online_weather || null}
            offlineMode={offlineMode}
            onTriggerSnapshot={handleTriggerSnapshot}
            onUploadFile={handleUploadFile}
            onRefreshAll={refreshAll}
            onNavigate={setPage}
            busy={busy}
          />
        )}
        {page === "radar" && (
          <RadarMapPage
            offlineMode={offlineMode}
            onToggleMode={handleToggleOfflineMode}
            onLocationSaved={refreshAll}
          />
        )}
        {page === "forecast" && <AnalyticsPage forecasts={forecasts} />}
        {page === "events" && <EventsPage events={events} onRefresh={refreshAll} />}
        {page === "ai" && <AiStudioPage />}
        {page === "settings" && <SettingsPage system={system} onConfigSaved={refreshAll} />}
      </main>

      {/* Mobile Bottom App Tab Bar (PWA-native feel) */}
      <nav className="sm:hidden fixed bottom-0 inset-x-0 z-40 border-t border-white/[0.09] bg-[#07090F]/95 backdrop-blur-xl px-2 py-1.5 flex items-center justify-around">
        {NAV_ITEMS.map(({ id, labelKey, icon: Icon }) => {
          const active = page === id;
          return (
            <button
              key={id}
              type="button"
              onClick={() => setPage(id)}
              className={`relative flex flex-col items-center gap-0.5 px-2 py-1 rounded-xl text-[10px] font-medium transition ${
                active ? "text-sky-300 bg-sky-400/15" : "text-[#8A96AA]"
              }`}
            >
              <Icon className="w-4 h-4" />
              <span className="truncate max-w-[54px]">{t(labelKey)}</span>
              {id === "events" && unackCount > 0 && (
                <span className="absolute top-0.5 right-1 w-2 h-2 rounded-full bg-amber-400" />
              )}
            </button>
          );
        })}
      </nav>

      <ManualModal open={manualOpen} currentPage={page} onClose={() => setManualOpen(false)} />

      {toast && (
        <div className="fixed bottom-20 sm:bottom-5 right-5 z-50 obs-glass-pill border-sky-400/30 px-4 py-2.5 text-xs text-sky-100 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-sky-400 shrink-0" />
          <span>{toast}</span>
        </div>
      )}
    </div>
  );
}

export default App;
