import React, { useEffect, useState } from "react";
import { Check, ChevronLeft, ChevronRight, Compass, Cpu, Globe, Lock, MapPin, RefreshCw, Video, X } from "lucide-react";
import { useI18n } from "../i18n/I18nContext";
import { api } from "../services/api";
import type { SystemStatus } from "../types/api";

interface SetupWizardModalProps {
  open: boolean;
  system: SystemStatus | null;
  onClose: () => void;
  onCompleted: (navigateTo?: string) => void;
}

const SOURCE_PRESETS = [
  { id: "webcam_device", labelEn: "USB / Virtual Webcam (Iriun)", labelIt: "Webcam USB / Virtuale (Iriun)", hint: "Device Index 0, 1..." },
  { id: "http_snapshot", labelEn: "Android IP Webcam (HTTP)", labelIt: "IP Webcam Android (HTTP)", hint: "http://192.168.x.x:8080/shot.jpg" },
  { id: "rtsp", labelEn: "RTSP IP Camera", labelIt: "Telecamera IP RTSP", hint: "rtsp://user:pass@192.168.x.x:554/stream1" },
  { id: "web_stream", labelEn: "Public Web Stream (HLS / SkylineWebcams)", labelIt: "Stream Web Pubblico (HLS / SkylineWebcams)", hint: "https://.../playlist.m3u8" },
  { id: "demo_simulation", labelEn: "Demo Sky Simulation", labelIt: "Simulazione Cielo Demo", hint: "No hardware needed" },
] as const;

const MOUNT_PRESETS = [
  { id: "zenith", labelEn: "Zenith (Straight Up 90°)", labelIt: "Zenit (Verticale 90°)", elevation: 90 },
  { id: "sky45", labelEn: "Tilted Sky (45°)", labelIt: "Cielo Inclinato (45°)", elevation: 45 },
  { id: "horizon", labelEn: "Panoramic Horizon (8°)", labelIt: "Orizzonte Panoramico (8°)", elevation: 8 },
] as const;

export const SetupWizardModal: React.FC<SetupWizardModalProps> = ({
  open,
  system,
  onClose,
  onCompleted,
}) => {
  const { locale, setLocale } = useI18n();
  const isIt = locale === "it";

  const [step, setStep] = useState<number>(1);
  const [offlineMode, setOfflineMode] = useState<boolean>(true);
  const [camName, setCamName] = useState<string>("Sky Camera 1");
  const [sourceType, setSourceType] = useState<string>("webcam_device");
  const [sourceUrl, setSourceUrl] = useState<string>("");
  const [deviceIndex, setDeviceIndex] = useState<number>(0);
  const [devices, setDevices] = useState<any[]>([]);
  const [testResult, setTestResult] = useState<any>(null);
  const [testing, setTesting] = useState<boolean>(false);

  const [locationQuery, setLocationQuery] = useState<string>("");
  const [geoResults, setGeoResults] = useState<any[]>([]);
  const [locationDesc, setLocationDesc] = useState<string>("");
  const [latitude, setLatitude] = useState<number>(41.9028);
  const [longitude, setLongitude] = useState<number>(12.4964);
  const [azimuthDeg, setAzimuthDeg] = useState<number>(180);
  const [elevationDeg, setElevationDeg] = useState<number>(55);
  const [fovH, setFovH] = useState<number>(74);
  const [tiltCorrection, setTiltCorrection] = useState<boolean>(true);

  const [hwProfile, setHwProfile] = useState<"auto" | "low_end" | "balanced" | "high_end" | "ultra">("auto");
  const [captureFirst, setCaptureFirst] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string>("");

  useEffect(() => {
    if (!open) return;
    setStep(1);
    setTestResult(null);
    setErrorMsg("");
    api
      .getSystemConfig()
      .then((cfg) => {
        if (!cfg) return;
        setOfflineMode(Boolean(cfg.server?.offline_mode ?? true));
        if (cfg.camera) {
          setCamName(cfg.camera.name || "Sky Camera 1");
          setSourceType(cfg.camera.source_type || "webcam_device");
          setSourceUrl(cfg.camera.source_url || "");
          setDeviceIndex(Number(cfg.camera.device_index ?? 0));
          setLocationDesc(cfg.camera.location_description || "");
          setLatitude(Number(cfg.camera.latitude ?? 41.9028));
          setLongitude(Number(cfg.camera.longitude ?? 12.4964));
          setAzimuthDeg(Number(cfg.camera.orientation_azimuth_deg ?? 180));
          setElevationDeg(Number(cfg.camera.orientation_elevation_deg ?? 55));
          setFovH(Number(cfg.camera.fov_horizontal_deg ?? 74));
          setTiltCorrection(Boolean(cfg.camera.tilt_correction_enabled ?? true));
        }
        if (cfg.ai?.hardware_profile && ["auto", "low_end", "balanced", "high_end", "ultra"].includes(cfg.ai.hardware_profile)) {
          setHwProfile(cfg.ai.hardware_profile);
        }
      })
      .catch(() => {});
  }, [open]);

  if (!open) return null;

  const scanDevices = async () => {
    try {
      const res = await api.listCameraDevices();
      setDevices(res.devices || []);
    } catch {
      setDevices([]);
    }
  };

  const testCameraConnection = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const res = await api.testCamera({
        source_type: sourceType,
        source_url: sourceUrl,
        device_index: deviceIndex,
      });
      setTestResult(res);
    } catch (e: any) {
      setTestResult({ ok: false, error: e?.message || "Connection test failed" });
    } finally {
      setTesting(false);
    }
  };

  const handleSearchCity = async () => {
    if (!locationQuery.trim()) return;
    try {
      const res = await api.searchLocation(locationQuery.trim(), locale);
      setGeoResults(res.results || []);
    } catch {
      setGeoResults([]);
    }
  };

  const handleUseBrowserGps = () => {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLatitude(Number(pos.coords.latitude.toFixed(5)));
        setLongitude(Number(pos.coords.longitude.toFixed(5)));
        if (!locationDesc) setLocationDesc("Local Station (GPS)");
      },
      () => {}
    );
  };

  const handleFinish = async (targetPage?: string) => {
    setSaving(true);
    setErrorMsg("");
    try {
      await api.updateHardwareProfile({ profile: hwProfile });
      await api.updateSystemConfig({
        server: {
          offline_mode: offlineMode,
          setup_completed: true,
        },
        camera: {
          name: camName.trim() || "Sky Camera 1",
          source_type: sourceType,
          source_url: sourceUrl.trim(),
          device_index: deviceIndex,
          location_description: locationDesc.trim(),
          latitude,
          longitude,
          orientation_azimuth_deg: azimuthDeg,
          orientation_elevation_deg: elevationDeg,
          fov_horizontal_deg: fovH,
          tilt_correction_enabled: tiltCorrection,
        },
      });
      if (captureFirst) {
        await api.triggerSnapshot().catch(() => {});
      }
      onCompleted(targetPage);
    } catch (e: any) {
      setErrorMsg(e?.message || "Failed to save configuration");
    } finally {
      setSaving(false);
    }
  };

  const steps = isIt
    ? [
        { id: 1, label: "1. Modalità & Lingua" },
        { id: 2, label: "2. Sorgente Cielo" },
        { id: 3, label: "3. Stazione & 3D" },
        { id: 4, label: "4. Hardware & Avvio" },
      ]
    : [
        { id: 1, label: "1. Mode & Locale" },
        { id: 2, label: "2. Camera Source" },
        { id: 3, label: "3. Station & 3D" },
        { id: 4, label: "4. Hardware & Launch" },
      ];

  return (
    <div className="fixed inset-0 z-50 bg-black/75 flex items-center justify-center p-4">
      <div className="w-full max-w-2xl obs-panel bg-[#16191F] border border-[#343B48] overflow-hidden shadow-2xl flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-5 py-3.5 bg-[#15181E] border-b border-[#262B34] flex items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-bold text-[#ECEFF4] tracking-wide uppercase">
              {isIt ? "Configurazione Guidata Stazione Meteo" : "Guided Meteorological Station Setup"}
            </h2>
            <p className="text-[11px] text-[#8D96A5]">
              {isIt
                ? "Configura modalità operativa, camera cielo, coordinate GPS e profilo hardware in 4 passaggi."
                : "Configure operation mode, sky camera source, station GPS coordinates, and hardware profile in 4 steps."}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close setup wizard"
            className="w-7 h-7 rounded bg-[#1B1F27] hover:bg-[#222731] border border-[#343B48] text-[#8D96A5] hover:text-[#ECEFF4] inline-flex items-center justify-center"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Stepper Bar */}
        <div className="grid grid-cols-4 divide-x divide-[#262B34] border-b border-[#262B34] bg-[#101216] text-[11px] font-mono">
          {steps.map((s) => (
            <button
              key={s.id}
              type="button"
              onClick={() => setStep(s.id)}
              className={`py-2 px-2 text-center truncate transition-colors ${
                step === s.id
                  ? "bg-[#1B1F27] text-[#ECEFF4] font-semibold shadow-[inset_0_-2px_0_0_#3B82F6]"
                  : step > s.id
                  ? "text-[#8D96A5] hover:text-[#ECEFF4]"
                  : "text-[#5D6675]"
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>

        {/* Step Body */}
        <div className="p-5 overflow-y-auto space-y-4 text-xs flex-1">
          {step === 1 && (
            <div className="space-y-4">
              <div className="space-y-1.5">
                <div className="obs-label">{isIt ? "Lingua Interfaccia" : "Interface Language"}</div>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setLocale("it")}
                    className={`p-2.5 rounded border text-left transition-colors ${
                      locale === "it"
                        ? "bg-[#1F2632] border-[#3B82F6] text-[#ECEFF4]"
                        : "bg-[#1B1F27] border-[#262B34] text-[#8D96A5] hover:text-[#ECEFF4]"
                    }`}
                  >
                    <div className="font-semibold">Italiano</div>
                    <div className="text-[11px] text-[#8D96A5]">Etichette meteo e manuale in italiano</div>
                  </button>
                  <button
                    type="button"
                    onClick={() => setLocale("en")}
                    className={`p-2.5 rounded border text-left transition-colors ${
                      locale === "en"
                        ? "bg-[#1F2632] border-[#3B82F6] text-[#ECEFF4]"
                        : "bg-[#1B1F27] border-[#262B34] text-[#8D96A5] hover:text-[#ECEFF4]"
                    }`}
                  >
                    <div className="font-semibold">English</div>
                    <div className="text-[11px] text-[#8D96A5]">International WMO / METAR interface</div>
                  </button>
                </div>
              </div>

              <div className="space-y-1.5">
                <div className="obs-label">{isIt ? "Modalità Operativa & Privacy" : "Operation Mode & Privacy"}</div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                  <button
                    type="button"
                    onClick={() => setOfflineMode(true)}
                    className={`p-3 rounded border text-left space-y-1 transition-colors ${
                      offlineMode
                        ? "bg-[#1F2632] border-[#3B82F6] text-[#ECEFF4]"
                        : "bg-[#1B1F27] border-[#262B34] text-[#8D96A5] hover:text-[#ECEFF4]"
                    }`}
                  >
                    <div className="font-semibold flex items-center gap-1.5">
                      <Lock className="w-3.5 h-3.5" />
                      {isIt ? "100% Locale Offline" : "100% Local Offline"}
                    </div>
                    <p className="text-[11px] text-[#8D96A5]">
                      {isIt
                        ? "Zero chiamate esterne. Analisi AI camera, nowcast +5/+15/+20m e altitudine stimata localmente."
                        : "Zero external requests. Local camera AI vision, +5/+15/+20m nowcast, and vision-estimated cloud base."}
                    </p>
                  </button>
                  <button
                    type="button"
                    onClick={() => setOfflineMode(false)}
                    className={`p-3 rounded border text-left space-y-1 transition-colors ${
                      !offlineMode
                        ? "bg-[#1F2632] border-[#3B82F6] text-[#ECEFF4]"
                        : "bg-[#1B1F27] border-[#262B34] text-[#8D96A5] hover:text-[#ECEFF4]"
                    }`}
                  >
                    <div className="font-semibold flex items-center gap-1.5">
                      <Globe className="w-3.5 h-3.5" />
                      {isIt ? "Ibrido Online (Open-Meteo + Radar)" : "Online Hybrid (Open-Meteo + Radar)"}
                    </div>
                    <p className="text-[11px] text-[#8D96A5]">
                      {isIt
                        ? "Integra gratuitamente Open-Meteo (T/Td, QNH, strati LCL) e radar precipitazioni RainViewer senza chiavi API."
                        : "Integrates free Open-Meteo NWP (T/Td, QNH, LCL cloud base) and RainViewer precipitation radar with zero API keys."}
                    </p>
                  </button>
                </div>
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="space-y-3.5">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <label className="space-y-1">
                  <span className="obs-label">{isIt ? "Nome Stazione / Camera" : "Station / Camera Name"}</span>
                  <input
                    type="text"
                    value={camName}
                    onChange={(e) => setCamName(e.target.value)}
                    className="obs-input w-full"
                  />
                </label>
                <label className="space-y-1">
                  <span className="obs-label">{isIt ? "Tipo Sorgente Video" : "Video Source Type"}</span>
                  <select
                    value={sourceType}
                    onChange={(e) => {
                      setSourceType(e.target.value);
                      setTestResult(null);
                    }}
                    className="obs-input w-full"
                  >
                    {SOURCE_PRESETS.map((p) => (
                      <option key={p.id} value={p.id}>
                        {isIt ? p.labelIt : p.labelEn}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              {sourceType === "webcam_device" ? (
                <div className="p-3 obs-inset space-y-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <label className="flex items-center gap-2">
                      <span className="text-[#8D96A5]">{isIt ? "Indice Dispositivo:" : "Device Index:"}</span>
                      <input
                        type="number"
                        min={0}
                        max={16}
                        value={deviceIndex}
                        onChange={(e) => setDeviceIndex(Number(e.target.value))}
                        className="obs-input w-20 font-mono tabular-nums"
                      />
                    </label>
                    <button type="button" onClick={scanDevices} className="obs-btn-secondary px-3 py-1.5">
                      {isIt ? "Scansiona Webcam" : "Scan Devices"}
                    </button>
                  </div>
                  {devices.length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                      {devices.map((d) => (
                        <button
                          key={d.index}
                          type="button"
                          onClick={() => setDeviceIndex(d.index)}
                          className={`px-2 py-1 rounded border font-mono text-[11px] ${
                            deviceIndex === d.index
                              ? "bg-[#1F2632] border-[#3B82F6] text-[#ECEFF4]"
                              : "bg-[#16191F] border-[#262B34] text-[#8D96A5]"
                          }`}
                        >
                          #{d.index} · {d.width}×{d.height}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ) : sourceType !== "demo_simulation" ? (
                <label className="block space-y-1">
                  <span className="obs-label">{isIt ? "URL Stream / Snapshot" : "Stream / Snapshot URL"}</span>
                  <input
                    type="text"
                    value={sourceUrl}
                    onChange={(e) => setSourceUrl(e.target.value)}
                    placeholder={SOURCE_PRESETS.find((p) => p.id === sourceType)?.hint || ""}
                    className="obs-input w-full font-mono"
                  />
                </label>
              ) : null}

              <div className="flex items-center gap-3">
                <button
                  type="button"
                  disabled={testing}
                  onClick={testCameraConnection}
                  className="obs-btn-secondary inline-flex items-center gap-1.5 px-3 py-1.5"
                >
                  <Video className="w-3.5 h-3.5 text-[#8D96A5]" />
                  {testing
                    ? isIt
                      ? "Verifica in corso..."
                      : "Testing connection..."
                    : isIt
                    ? "Testa Connessione Camera"
                    : "Test Camera Connection"}
                </button>
                {testResult && (
                  <span className={`font-mono text-[11px] ${testResult.ok ? "text-emerald-400" : "text-rose-400"}`}>
                    {testResult.ok
                      ? `OK · ${testResult.width}×${testResult.height} · ${testResult.latency_ms} ms`
                      : testResult.error || "Failed"}
                  </span>
                )}
              </div>
            </div>
          )}

          {step === 3 && (
            <div className="space-y-3.5">
              <div className="space-y-1.5">
                <span className="obs-label">{isIt ? "Cerca Città o Usa GPS" : "Search City or Use GPS"}</span>
                <div className="flex flex-wrap gap-2">
                  <input
                    type="text"
                    value={locationQuery}
                    onChange={(e) => setLocationQuery(e.target.value)}
                    placeholder={isIt ? "Es. Milano, Roma, Bologna..." : "e.g. Rome, Zurich, London..."}
                    className="obs-input flex-1 min-w-[180px]"
                  />
                  {!offlineMode && (
                    <button type="button" onClick={handleSearchCity} className="obs-btn-secondary px-3 py-1.5">
                      {isIt ? "Cerca" : "Search"}
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={handleUseBrowserGps}
                    className="obs-btn-secondary inline-flex items-center gap-1 px-3 py-1.5"
                  >
                    <MapPin className="w-3.5 h-3.5 text-[#8D96A5]" />
                    GPS
                  </button>
                </div>
                {geoResults.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 pt-1">
                    {geoResults.slice(0, 5).map((g, i) => (
                      <button
                        key={i}
                        type="button"
                        onClick={() => {
                          setLatitude(Number(g.latitude));
                          setLongitude(Number(g.longitude));
                          setLocationDesc(g.display_name || g.name);
                          setGeoResults([]);
                        }}
                        className="px-2 py-1 rounded bg-[#1B1F27] border border-[#343B48] hover:border-[#3B82F6] text-[11px] text-[#ECEFF4]"
                      >
                        {g.display_name || g.name}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                <label className="space-y-1">
                  <span className="obs-label">{isIt ? "Nome Località" : "Location Label"}</span>
                  <input
                    type="text"
                    value={locationDesc}
                    onChange={(e) => setLocationDesc(e.target.value)}
                    placeholder="Roma, IT"
                    className="obs-input w-full"
                  />
                </label>
                <label className="space-y-1">
                  <span className="obs-label">Latitude</span>
                  <input
                    type="number"
                    step="0.0001"
                    value={latitude}
                    onChange={(e) => setLatitude(Number(e.target.value))}
                    className="obs-input w-full font-mono tabular-nums"
                  />
                </label>
                <label className="space-y-1">
                  <span className="obs-label">Longitude</span>
                  <input
                    type="number"
                    step="0.0001"
                    value={longitude}
                    onChange={(e) => setLongitude(Number(e.target.value))}
                    className="obs-input w-full font-mono tabular-nums"
                  />
                </label>
              </div>

              <div className="space-y-1.5">
                <span className="obs-label flex items-center gap-1.5">
                  <Compass className="w-3.5 h-3.5 text-[#8D96A5]" />
                  {isIt ? "Orientamento & Inclinazione Camera 3D" : "3D Camera Orientation & Tilt"}
                </span>
                <div className="grid grid-cols-3 gap-2">
                  {MOUNT_PRESETS.map((mp) => (
                    <button
                      key={mp.id}
                      type="button"
                      onClick={() => setElevationDeg(mp.elevation)}
                      className={`p-2 rounded border text-left text-[11px] transition-colors ${
                        Math.abs(elevationDeg - mp.elevation) <= 3
                          ? "bg-[#1F2632] border-[#3B82F6] text-[#ECEFF4]"
                          : "bg-[#1B1F27] border-[#262B34] text-[#8D96A5]"
                      }`}
                    >
                      {isIt ? mp.labelIt : mp.labelEn}
                    </button>
                  ))}
                </div>
                <div className="grid grid-cols-3 gap-2.5 pt-1">
                  <label className="space-y-1">
                    <span className="text-[11px] text-[#8D96A5]">{isIt ? "Azimut (0=N, 180=S)" : "Azimuth (0=N)"}</span>
                    <input
                      type="number"
                      min={0}
                      max={360}
                      value={azimuthDeg}
                      onChange={(e) => setAzimuthDeg(Number(e.target.value))}
                      className="obs-input w-full font-mono tabular-nums"
                    />
                  </label>
                  <label className="space-y-1">
                    <span className="text-[11px] text-[#8D96A5]">{isIt ? "Elevazione (-20..90°)" : "Elevation (-20..90°)"}</span>
                    <input
                      type="number"
                      min={-20}
                      max={90}
                      value={elevationDeg}
                      onChange={(e) => setElevationDeg(Number(e.target.value))}
                      className="obs-input w-full font-mono tabular-nums"
                    />
                  </label>
                  <label className="space-y-1">
                    <span className="text-[11px] text-[#8D96A5]">FOV H (°)</span>
                    <input
                      type="number"
                      min={20}
                      max={175}
                      value={fovH}
                      onChange={(e) => setFovH(Number(e.target.value))}
                      className="obs-input w-full font-mono tabular-nums"
                    />
                  </label>
                </div>
                <label className="flex items-center gap-2 pt-1 text-[#CBD5E1] cursor-pointer">
                  <input
                    type="checkbox"
                    checked={tiltCorrection}
                    onChange={(e) => setTiltCorrection(e.target.checked)}
                  />
                  <span>
                    {isIt
                      ? "Abilita correzione prospettica 3D, taglio orizzonte e rilevamento aureola solare"
                      : "Enable 3D tilt perspective correction, horizon cut, and solar aureole detection"}
                  </span>
                </label>
              </div>
            </div>
          )}

          {step === 4 && (
            <div className="space-y-3.5">
              <div className="p-3 obs-inset flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-[#8D96A5]" />
                  <div>
                    <div className="font-semibold text-[#ECEFF4]">
                      {isIt ? "Hardware Rilevato" : "Detected Hardware"}
                    </div>
                    <div className="text-[11px] font-mono text-[#8D96A5]">
                      {system
                        ? `${system.hardware.cpu_cores} CPU · ${system.hardware.ram_total_gb} GB RAM · GPU: ${system.hardware.gpu.active_device.toUpperCase()} (${system.hardware.gpu.device_name})`
                        : "Detecting..."}
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-1.5">
                <span className="obs-label">{isIt ? "Profilo Prestazioni AI" : "AI Performance Profile"}</span>
                <div className="grid grid-cols-5 gap-1.5">
                  {(["auto", "low_end", "balanced", "high_end", "ultra"] as const).map((p) => (
                    <button
                      key={p}
                      type="button"
                      onClick={() => setHwProfile(p)}
                      className={`py-2 px-1.5 rounded border text-center font-mono text-[11px] uppercase transition-colors ${
                        hwProfile === p
                          ? "bg-[#1F2632] border-[#3B82F6] text-[#ECEFF4] font-semibold"
                          : "bg-[#1B1F27] border-[#262B34] text-[#8D96A5]"
                      }`}
                    >
                      {p.replace("_", " ")}
                    </button>
                  ))}
                </div>
              </div>

              <div className="border border-[#262B34] rounded divide-y divide-[#262B34] text-[11px] font-mono">
                <div className="px-3 py-1.5 flex justify-between">
                  <span className="text-[#8D96A5]">{isIt ? "Modalità" : "Mode"}</span>
                  <span className="text-[#ECEFF4]">{offlineMode ? "Offline Local" : "Online Hybrid (Open-Meteo)"}</span>
                </div>
                <div className="px-3 py-1.5 flex justify-between">
                  <span className="text-[#8D96A5]">{isIt ? "Camera" : "Camera"}</span>
                  <span className="text-[#ECEFF4]">{camName} ({sourceType})</span>
                </div>
                <div className="px-3 py-1.5 flex justify-between">
                  <span className="text-[#8D96A5]">{isIt ? "Coordinate & Geometria" : "Coords & Geometry"}</span>
                  <span className="text-[#ECEFF4]">
                    {latitude.toFixed(3)}°, {longitude.toFixed(3)}° · El {elevationDeg}° · Az {azimuthDeg}°
                  </span>
                </div>
              </div>

              <label className="flex items-center gap-2 text-[#CBD5E1] cursor-pointer">
                <input
                  type="checkbox"
                  checked={captureFirst}
                  onChange={(e) => setCaptureFirst(e.target.checked)}
                />
                <span>
                  {isIt
                    ? "Acquisisci subito la prima osservazione cielo al termine"
                    : "Capture the first sky observation immediately upon finish"}
                </span>
              </label>

              {errorMsg && <div className="text-rose-400 font-mono text-[11px]">{errorMsg}</div>}
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className="px-5 py-3 bg-[#15181E] border-t border-[#262B34] flex items-center justify-between gap-2 text-xs">
          <button
            type="button"
            disabled={step === 1 || saving}
            onClick={() => setStep((s) => Math.max(1, s - 1))}
            className="obs-btn-secondary inline-flex items-center gap-1 px-3 py-1.5"
          >
            <ChevronLeft className="w-3.5 h-3.5" />
            {isIt ? "Indietro" : "Back"}
          </button>

          <div className="flex items-center gap-2">
            {step < 4 ? (
              <button
                type="button"
                onClick={() => setStep((s) => Math.min(4, s + 1))}
                className="obs-btn-primary inline-flex items-center gap-1 px-4 py-1.5"
              >
                {isIt ? "Avanti" : "Next"}
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            ) : (
              <>
                <button
                  type="button"
                  disabled={saving}
                  onClick={() => handleFinish("settings")}
                  className="obs-btn-secondary px-3 py-1.5"
                >
                  {isIt ? "Salva e Dipingi Maschera" : "Save & Paint Mask"}
                </button>
                <button
                  type="button"
                  disabled={saving}
                  onClick={() => handleFinish("dashboard")}
                  className="obs-btn-primary inline-flex items-center gap-1.5 px-4 py-1.5"
                >
                  {saving ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Check className="w-3.5 h-3.5" />}
                  {isIt ? "Completa e Avvia" : "Finish & Launch"}
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
