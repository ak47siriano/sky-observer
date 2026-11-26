import { useCallback, useEffect, useState } from "react";
import {
  Bell,
  CloudSun,
  Cpu,
  Globe,
  HelpCircle,
  Languages,
  LayoutDashboard,
  Lock,
  Map as MapIcon,
  MapPin,
  Settings,
  SlidersHorizontal,
} from "lucide-react";
import { ManualModal } from "./components/ManualModal";
import { SetupWizardModal } from "./components/SetupWizardModal";
import { useI18n } from "./i18n/I18nContext";
import { api } from "./services/api";
import type { EventAlert, ForecastItem, Observation, SystemStatus } from "./types/api";
import { AiStudioPage } from "./pages/AiStudioPage";
import { AnalyticsPage } from "./pages/AnalyticsPage";
import { DashboardPage } from "./pages/DashboardPage";
import { EventsPage } from "./pages/EventsPage";
import { RadarMapPage } from "./pages/RadarMapPage";
import { SettingsPage } from "./pages/SettingsPage";
import { SimpleDashboardPage } from "./pages/SimpleDashboardPage";
import { SimpleMapPage } from "./pages/SimpleMapPage";
import { SimpleSettingsPage } from "./pages/SimpleSettingsPage";

const EXPERT_NAV_ITEMS = [
  { id: "dashboard", labelKey: "nav.dashboard", icon: LayoutDashboard },
  { id: "radar", labelKey: "nav.radar", icon: MapPin },
  { id: "forecast", labelKey: "nav.forecast", icon: CloudSun },
  { id: "events", labelKey: "nav.events", icon: Bell },
  { id: "ai", labelKey: "nav.ai", icon: Cpu },
  { id: "settings", labelKey: "nav.settings", icon: Settings },
] as const;

const SIMPLE_NAV_ITEMS = [
  { id: "simple_sky", labelKey: "simple.nav.sky", icon: CloudSun },
  { id: "simple_map", labelKey: "simple.nav.map", icon: MapIcon },
  { id: "simple_base", labelKey: "simple.nav.base", icon: SlidersHorizontal },
] as const;

type UiMode = "simple" | "expert";

const UI_MODE_STORAGE_KEY = "sky_observer_ui_mode";

export function App() {
  const { locale, setLocale, t } = useI18n();
  const [uiMode, setUiMode] = useState<UiMode>(() => {
    const qMode = new URLSearchParams(window.location.search).get("mode");
    if (qMode === "simple" || qMode === "expert") return qMode;
    try {
      const saved = window.localStorage.getItem(UI_MODE_STORAGE_KEY);
      if (saved === "simple" || saved === "expert") return saved;
    } catch {
      // ignore storage errors
    }
    return "expert";
  });

  const [page, setPage] = useState<string>(() => {
    const q = new URLSearchParams(window.location.search).get("page");
    return q && ["dashboard", "radar", "forecast", "events", "ai", "settings"].includes(q) ? q : "dashboard";
  });
  const [simplePage, setSimplePage] = useState<"simple_sky" | "simple_map" | "simple_base">("simple_sky");

  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [latestObs, setLatestObs] = useState<Observation | null>(null);
  const [forecasts, setForecasts] = useState<ForecastItem[]>([]);
  const [events, setEvents] = useState<EventAlert[]>([]);
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState<string>("");
  const [manualOpen, setManualOpen] = useState(false);
  const [wizardOpen, setWizardOpen] = useState(false);
  const [wizardAutoDismissed, setWizardAutoDismissed] = useState(false);
  const [utcTime, setUtcTime] = useState<string>(() => new Date().toISOString().slice(11, 19));

  const switchUiMode = (next: UiMode) => {
    setUiMode(next);
    try {
      window.localStorage.setItem(UI_MODE_STORAGE_KEY, next);
    } catch {
      // ignore storage errors
    }
    // Keep natural context when switching between Dashboard <-> Sky Now, Map <-> Map, Settings <-> Base
    if (next === "simple") {
      if (page === "radar") setSimplePage("simple_map");
      else if (page === "settings") setSimplePage("simple_base");
      else setSimplePage("simple_sky");
    } else {
      if (simplePage === "simple_map") setPage("radar");
      else if (simplePage === "simple_base") setPage("settings");
      else setPage("dashboard");
    }
  };

  useEffect(() => {
    const iconHref = uiMode === "simple" ? "/logo-simple.svg" : "/logo-expert.svg";
    const link = document.querySelector<HTMLLinkElement>("link[rel='icon']");
    if (link) {
      link.type = "image/svg+xml";
      link.href = iconHref;
    }
  }, [uiMode]);

  useEffect(() => {
    if (
      system &&
      system.setup_completed === false &&
      (system.observations_count ?? 0) === 0 &&
      !system.camera?.connected &&
      !wizardAutoDismissed
    ) {
      setWizardOpen(true);
    }
  }, [system, wizardAutoDismissed]);

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
  const isSimple = uiMode === "simple";

  return (
    <div className="min-h-screen text-[#ECEFF4] flex flex-col pb-16 sm:pb-0">
      {/* Top Bar — Shared Brand + Always-Visible [ Simple | Expert ] Switch */}
      <header className="sticky top-0 z-40 border-b border-[#262B34] bg-[#15181E]">
        <div className="max-w-[1800px] w-full mx-auto px-3 sm:px-4 h-14 py-2 flex items-center justify-between gap-3">
          {/* Brand */}
          <button
            type="button"
            onClick={() => (isSimple ? setSimplePage("simple_sky") : setPage("dashboard"))}
            className="flex items-center gap-2.5 shrink-0 text-left"
          >
            <div className="relative shrink-0">
              <img
                src={isSimple ? "/logo-simple.svg" : "/logo-expert.svg"}
                alt={isSimple ? "Sky Observer Simple" : "Sky Observer Expert"}
                className="w-10 h-10 object-contain transition-transform duration-200 hover:scale-105"
              />
              <span
                className={`absolute bottom-0 right-0 w-2.5 h-2.5 rounded-full ring-2 ring-[#15181E] ${
                  system?.camera.connected ? "bg-emerald-500" : "bg-rose-500"
                }`}
              />
            </div>
            <div className="hidden sm:block leading-tight">
              <div className="text-[13px] font-bold tracking-wider text-[#ECEFF4] flex items-center gap-1.5">
                <span>SKY OBSERVER</span>
                {!isSimple && latestObs && (
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
              {!isSimple && (
                <div className="text-[10px] font-mono text-[#8D96A5] tracking-wide">
                  {offlineMode ? "LOCAL · OFFLINE" : "HYBRID · OPEN-METEO"}
                </div>
              )}
            </div>
          </button>

          {/* Center Navigation: 3 Friendly Pills in Simple Mode vs 6 Full Tabs in Expert Mode */}
          {isSimple ? (
            <nav className="hidden sm:flex items-center gap-1.5 p-1 rounded-full bg-[#101216] border border-white/[0.09]">
              {SIMPLE_NAV_ITEMS.map(({ id, labelKey, icon: Icon }) => {
                const active = simplePage === id;
                return (
                  <button
                    key={id}
                    type="button"
                    onClick={() => setSimplePage(id)}
                    className={`inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs sm:text-sm font-semibold transition ${
                      active
                        ? "bg-sky-400 text-[#07131D] shadow"
                        : "text-[#CBD5E1] hover:text-white hover:bg-white/[0.05]"
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{t(labelKey)}</span>
                  </button>
                );
              })}
            </nav>
          ) : (
            <nav className="hidden sm:flex items-center gap-1 p-0.5 rounded bg-[#101216] border border-[#262B34] overflow-x-auto no-scrollbar">
              {EXPERT_NAV_ITEMS.map(({ id, labelKey, icon: Icon }) => {
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
          )}

          {/* Right Controls: ALWAYS-PRESENT [ Semplice | Expert ] Switch + Mode-Appropriate Status */}
          <div className="flex items-center gap-2 shrink-0 text-[11px]">
            {/* Persistent Mode Switch Pill */}
            <div
              role="group"
              aria-label={t("uiMode.ariaLabel")}
              className="inline-flex items-center p-0.5 rounded-full bg-[#101216] border border-[#343B48] shadow-inner"
            >
              <button
                type="button"
                onClick={() => switchUiMode("simple")}
                className={`px-3 py-1 rounded-full text-[11px] sm:text-xs font-bold transition ${
                  isSimple
                    ? "bg-gradient-to-r from-sky-400 to-emerald-400 text-[#06181A] shadow"
                    : "text-[#9AA1AD] hover:text-white"
                }`}
              >
                {t("uiMode.simple")}
              </button>
              <button
                type="button"
                onClick={() => switchUiMode("expert")}
                className={`px-3 py-1 rounded-full text-[11px] sm:text-xs font-bold transition ${
                  !isSimple
                    ? "bg-[#2563EB] text-white shadow"
                    : "text-[#9AA1AD] hover:text-white"
                }`}
              >
                {t("uiMode.expert")}
              </button>
            </div>

            {isSimple ? (
              <div className="hidden md:inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[#101216] border border-white/[0.08] text-xs font-semibold text-[#ECEFF4]">
                <span
                  className={`w-2 h-2 rounded-full ${
                    system?.camera.connected ? "bg-emerald-400" : "bg-rose-400"
                  }`}
                />
                <span>{system?.camera.connected ? t("simple.camOk") : t("simple.camOff")}</span>
              </div>
            ) : (
              <>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => handleToggleOfflineMode(!offlineMode)}
                  title="Toggle Offline Only vs Online Hybrid (Open-Meteo + RainViewer)"
                  className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded border border-[#343B48] bg-[#1B1F27] hover:bg-[#222731] text-[#ECEFF4] font-medium transition-colors"
                >
                  {offlineMode ? (
                    <Lock className="w-3 h-3 text-[#8D96A5]" />
                  ) : (
                    <Globe className="w-3 h-3 text-[#8D96A5]" />
                  )}
                  <span className="hidden xl:inline">
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
                  className="hidden xl:inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded bg-[#101216] border border-[#262B34] font-mono tabular-nums text-[#8D96A5]"
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

                <button
                  type="button"
                  onClick={() => setManualOpen(true)}
                  title={locale === "it" ? "Apri il Manuale (sezione pagina corrente)" : "Open Manual (current page section)"}
                  aria-label={locale === "it" ? "Apri il Manuale" : "Open Manual"}
                  className="w-8 h-8 rounded bg-[#1B1F27] hover:bg-[#222731] border border-[#343B48] text-[#8D96A5] hover:text-[#ECEFF4] inline-flex items-center justify-center transition-colors"
                >
                  <HelpCircle className="w-4 h-4" />
                </button>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 w-full max-w-[1800px] mx-auto px-3 sm:px-4 py-3">
        {isSimple ? (
          <>
            {simplePage === "simple_sky" && (
              <SimpleDashboardPage
                camera={system?.camera || null}
                obs={latestObs}
                forecasts={forecasts}
                events={events}
                onlineWeather={system?.online_weather || null}
                offlineMode={offlineMode}
                onTriggerSnapshot={handleTriggerSnapshot}
                onUploadFile={handleUploadFile}
                onOpenSimpleMap={() => setSimplePage("simple_map")}
                onOpenSimpleBase={() => setSimplePage("simple_base")}
                busy={busy}
              />
            )}
            {simplePage === "simple_map" && (
              <SimpleMapPage offlineMode={offlineMode} onToggleMode={handleToggleOfflineMode} />
            )}
            {simplePage === "simple_base" && (
              <SimpleSettingsPage
                offlineMode={offlineMode}
                onToggleMode={handleToggleOfflineMode}
                onConfigSaved={refreshAll}
                onSwitchToExpert={() => switchUiMode("expert")}
                onOpenSetupWizard={() => setWizardOpen(true)}
              />
            )}
          </>
        ) : (
          <>
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
            {page === "settings" && (
              <SettingsPage
                system={system}
                onConfigSaved={refreshAll}
                onOpenSetupWizard={() => setWizardOpen(true)}
              />
            )}
          </>
        )}
      </main>

      {/* Mobile Bottom App Tab Bar */}
      <nav className="sm:hidden fixed bottom-0 inset-x-0 z-40 border-t border-white/[0.09] bg-[#07090F]/95 backdrop-blur-xl px-2 py-1.5 flex items-center justify-around">
        {isSimple
          ? SIMPLE_NAV_ITEMS.map(({ id, labelKey, icon: Icon }) => {
              const active = simplePage === id;
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => setSimplePage(id)}
                  className={`flex flex-col items-center gap-0.5 px-4 py-1.5 rounded-xl text-[11px] font-semibold transition ${
                    active ? "text-sky-300 bg-sky-400/15" : "text-[#8A96AA]"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{t(labelKey)}</span>
                </button>
              );
            })
          : EXPERT_NAV_ITEMS.map(({ id, labelKey, icon: Icon }) => {
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

      <SetupWizardModal
        open={wizardOpen}
        system={system}
        onClose={() => {
          setWizardOpen(false);
          setWizardAutoDismissed(true);
        }}
        onCompleted={(targetPage) => {
          setWizardOpen(false);
          setWizardAutoDismissed(true);
          refreshAll();
          if (isSimple) {
            setSimplePage("simple_sky");
          } else if (targetPage) {
            setPage(targetPage);
          }
          showToast(locale === "it" ? "Configurazione stazione completata" : "Station setup completed");
        }}
      />

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
