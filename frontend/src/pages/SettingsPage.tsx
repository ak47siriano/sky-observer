import React, { useCallback, useEffect, useState } from "react";
import { AlertTriangle, CheckCircle, Database, HardDrive, Plus, RotateCcw, Save, Shield, Trash2, Wifi } from "lucide-react";
import { CameraTiltCalibrator, type CameraGeometryValues } from "../components/CameraTiltCalibrator";
import { ExclusionMaskPainter } from "../components/ExclusionMaskPainter";
import { useI18n } from "../i18n/I18nContext";
import { api } from "../services/api";
import type { SystemStatus } from "../types/api";

const PURGE_CATEGORIES = [
  { id: "raw_images", color: "#38BDF8" },
  { id: "processed_images", color: "#818CF8" },
  { id: "observations", color: "#22D3EE" },
  { id: "forecasts", color: "#34D399" },
  { id: "events", color: "#FBBF24" },
  { id: "timelapse", color: "#A78BFA" },
  { id: "map_tiles", color: "#60A5FA" },
  { id: "exclusion_masks", color: "#FB7185" },
  { id: "datasets", color: "#F472B6" },
  { id: "models", color: "#F43F5E" },
  { id: "reset_config", color: "#94A3B8" },
] as const;

export const SettingsPage: React.FC<{
  system: SystemStatus | null;
  onConfigSaved: () => void;
  onOpenSetupWizard?: () => void;
}> = ({
  system,
  onConfigSaved,
  onOpenSetupWizard,
}) => {
  const { t, tEnum, locale } = useI18n();
  const [cfg, setCfg] = useState<any>(null);
  const [devices, setDevices] = useState<any[]>([]);
  const [testResult, setTestResult] = useState<any>(null);
  const [saveMsg, setSaveMsg] = useState("");
  const [applyingGeom, setApplyingGeom] = useState(false);
  const [newPolyPts, setNewPolyPts] = useState<number[][]>([]);
  const [polyCategory, setPolyCategory] = useState<
    "trees" | "buildings" | "poles" | "roofs" | "road" | "logo" | "other"
  >("trees");
  const [polyLabel, setPolyLabel] = useState("Obstruction");
  const [storageInfo, setStorageInfo] = useState<any>(null);
  const [secondaryCams, setSecondaryCams] = useState<any[]>([]);
  const [multiCamBusy, setMultiCamBusy] = useState(false);
  const [selectedTargets, setSelectedTargets] = useState<string[]>([
    "raw_images",
    "processed_images",
    "observations",
    "forecasts",
    "timelapse",
  ]);
  const [preserveTraining, setPreserveTraining] = useState(true);
  const [preserveCam, setPreserveCam] = useState(true);
  const [pendingPurgeTargets, setPendingPurgeTargets] = useState<string[] | null>(null);
  const [purging, setPurging] = useState(false);
  const [purgeMsg, setPurgeMsg] = useState("");

  const refreshStorage = useCallback(() => {
    api.getStorageStats().then(setStorageInfo).catch(() => {});
  }, []);

  useEffect(() => {
    api.getSystemConfig().then(setCfg).catch(() => {});
    api.getSecondaryCameras().then((d) => setSecondaryCams(d.cameras || [])).catch(() => {});
    refreshStorage();
  }, [refreshStorage]);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!cfg) return;
    await api.updateSystemConfig(cfg);
    setSaveMsg(t("settings.savedMsg"));
    onConfigSaved();
    setTimeout(() => setSaveMsg(""), 3000);
  };

  const handleTestCam = async () => {
    if (!cfg) return;
    const res = await api.testCamera({
      source_type: cfg.camera.source_type,
      source_url: cfg.camera.source_url,
      device_index: cfg.camera.device_index,
    });
    setTestResult(res);
  };

  const handleScanDevices = async () => {
    const res = await api.listCameraDevices();
    setDevices(res.devices || []);
  };

  const addExclusionPolygon = async () => {
    if (!cfg || newPolyPts.length < 3) return;
    const updatedPolys = [
      ...(cfg.camera.exclusion_polygons || []),
      {
        id: `poly_${Date.now()}`,
        label: polyLabel || polyCategory,
        category: polyCategory,
        points: newPolyPts,
      },
    ];
    const nextCam = await api.updateMasks({
      sky_roi_polygon: cfg.camera.sky_roi_polygon || [],
      exclusion_polygons: updatedPolys,
    });
    setCfg({ ...cfg, camera: nextCam });
    setNewPolyPts([]);
    onConfigSaved();
  };

  const removeExclusionPolygon = async (id: string) => {
    if (!cfg) return;
    const updatedPolys = (cfg.camera.exclusion_polygons || []).filter((p: any) => p.id !== id);
    const nextCam = await api.updateMasks({
      sky_roi_polygon: cfg.camera.sky_roi_polygon || [],
      exclusion_polygons: updatedPolys,
    });
    setCfg({ ...cfg, camera: nextCam });
    onConfigSaved();
  };

  const setAsSkyRoi = async () => {
    if (!cfg || newPolyPts.length < 3) return;
    const nextCam = await api.updateMasks({
      sky_roi_polygon: newPolyPts,
      exclusion_polygons: cfg.camera.exclusion_polygons || [],
    });
    setCfg({ ...cfg, camera: nextCam });
    setNewPolyPts([]);
    onConfigSaved();
  };

  const handleApplyGeometry = async () => {
    if (!cfg) return;
    setApplyingGeom(true);
    try {
      await api.updateCameraGeometry({
        orientation_elevation_deg: Number(cfg.camera.orientation_elevation_deg ?? 65),
        orientation_azimuth_deg: Number(cfg.camera.orientation_azimuth_deg ?? 0),
        orientation_roll_deg: Number(cfg.camera.orientation_roll_deg ?? 0),
        fov_horizontal_deg: Number(cfg.camera.fov_horizontal_deg ?? 72),
        fov_vertical_deg: Number(cfg.camera.fov_vertical_deg ?? 45),
        assumed_cloud_altitude_m: cfg.camera.assumed_cloud_altitude_m ?? null,
        tilt_correction_enabled: Boolean(cfg.camera.tilt_correction_enabled ?? true),
        recalculate_latest: true,
      });
      setSaveMsg(t("tilt.appliedMsg"));
      onConfigSaved();
      setTimeout(() => setSaveMsg(""), 3500);
    } finally {
      setApplyingGeom(false);
    }
  };

  const togglePurgeTarget = (id: string) => {
    setSelectedTargets((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const executePurge = async (targets: string[]) => {
    if (!targets.length) return;
    setPurging(true);
    try {
      const res = await api.purgeStorage({
        targets,
        preserve_training_samples: preserveTraining && !targets.includes("datasets"),
        preserve_camera_connection: preserveCam,
      });
      setStorageInfo(res.storage);
      setPendingPurgeTargets(null);
      const p = res.purged || {};
      const totalFiles =
        Number(p.deleted_raw_files || 0) +
        Number(p.deleted_processed_files || 0) +
        Number(p.deleted_timelapse_files || 0) +
        Number(p.deleted_map_tiles || 0) +
        Number(p.deleted_exclusion_masks || 0);
      setPurgeMsg(
        t("data.purgedSuccess", {
          obs: p.deleted_observations ?? 0,
          files: totalFiles,
        })
      );
      const freshCfg = await api.getSystemConfig();
      setCfg(freshCfg);
      onConfigSaved();
      setTimeout(() => setPurgeMsg(""), 5000);
    } finally {
      setPurging(false);
    }
  };

  if (!cfg) {
    return <div className="p-8 text-center text-sm text-[#8E95A2]">{t("settings.loading")}</div>;
  }

  return (
    <div className="space-y-4">
      <form onSubmit={handleSave} className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-white">{t("settings.title")}</h2>
            <p className="text-xs text-[#8E95A2]">{t("settings.subtitle")}</p>
          </div>
          <div className="flex items-center gap-2.5">
            {saveMsg && (
              <span className="inline-flex items-center gap-1 text-xs text-emerald-400">
                <CheckCircle className="w-3.5 h-3.5" /> {saveMsg}
              </span>
            )}
            {onOpenSetupWizard && (
              <button
                type="button"
                onClick={onOpenSetupWizard}
                className="obs-btn-secondary inline-flex items-center gap-1.5 px-3 py-2 text-xs"
              >
                {locale === "it" ? "Configurazione Guidata" : "Guided Setup Wizard"}
              </button>
            )}
            <button
              type="submit"
              className="obs-btn-primary inline-flex items-center gap-1.5 px-4 py-2 text-xs"
            >
              <Save className="w-3.5 h-3.5" /> {t("settings.saveAll")}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 text-xs">
          <div className="obs-panel p-5 space-y-3">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Wifi className="w-4 h-4 text-sky-400" /> {t("settings.camTitle")}
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <label className="space-y-1">
                <span className="text-[#9AA1AD]">{t("settings.camName")}</span>
                <input
                  type="text"
                  value={cfg.camera.name}
                  onChange={(e) => setCfg({ ...cfg, camera: { ...cfg.camera, name: e.target.value } })}
                  className="obs-input w-full"
                />
              </label>
              <label className="space-y-1">
                <span className="text-[#9AA1AD]">{t("settings.sourceType")}</span>
                <select
                  value={cfg.camera.source_type}
                  onChange={(e) => setCfg({ ...cfg, camera: { ...cfg.camera, source_type: e.target.value } })}
                  className="obs-input w-full"
                >
                  <option value="http_snapshot">{t("settings.src.http")}</option>
                  <option value="mjpeg_stream">{t("settings.src.mjpeg")}</option>
                  <option value="rtsp">{t("settings.src.rtsp")}</option>
                  <option value="web_stream">{t("settings.src.web_stream")}</option>
                  <option value="webcam_device">{t("settings.src.webcam")}</option>
                  <option value="allsky_folder">{t("settings.src.allsky")}</option>
                  <option value="local_import">{t("settings.src.local")}</option>
                  <option value="demo_simulation">{t("settings.src.demo")}</option>
                </select>
              </label>
            </div>

            {cfg.camera.source_type === "webcam_device" ? (
              <div className="space-y-1.5">
                <span className="text-[#9AA1AD]">{t("settings.deviceIndexLabel")}</span>
                <div className="flex flex-wrap gap-2">
                  <input
                    type="number"
                    min={0}
                    max={16}
                    value={cfg.camera.device_index ?? 0}
                    onChange={(e) =>
                      setCfg({ ...cfg, camera: { ...cfg.camera, device_index: Number(e.target.value) } })
                    }
                    className="obs-input w-24 font-mono tabular-nums"
                  />
                  <button type="button" onClick={handleScanDevices} className="obs-btn-secondary px-3 py-1.5">
                    {t("settings.scanDevices")}
                  </button>
                  <button type="button" onClick={handleTestCam} className="obs-btn-primary px-3 py-1.5">
                    {t("settings.testDevice")}
                  </button>
                </div>
                {devices.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 pt-1">
                    {devices.map((d) => (
                      <button
                        key={d.index}
                        type="button"
                        onClick={() => setCfg({ ...cfg, camera: { ...cfg.camera, device_index: d.index } })}
                        className={`px-2 py-1 rounded-lg border font-mono text-[11px] ${
                          cfg.camera.device_index === d.index
                            ? "bg-white/[0.12] border-white/30 text-white"
                            : "bg-[#0F1116] border-white/[0.08] text-[#9AA1AD]"
                        }`}
                      >
                        {t("settings.indexOption", { idx: d.index, w: d.width, h: d.height, backend: d.backend })}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <label className="block space-y-1">
                <span className="text-[#9AA1AD]">{t("settings.camUrl")}</span>
                <div className="flex gap-2">
                  <input
                    type="text"
                    placeholder="http://192.168.1.50:8080/shot.jpg or rtsp://192.168.1.50:8554/live"
                    value={cfg.camera.source_url}
                    onChange={(e) => setCfg({ ...cfg, camera: { ...cfg.camera, source_url: e.target.value } })}
                    className="obs-input flex-1 font-mono"
                  />
                  <button type="button" onClick={handleTestCam} className="obs-btn-secondary px-3 py-1.5">
                    {t("settings.test")}
                  </button>
                </div>
                {cfg.camera.source_type === "web_stream" && (
                  <p className="text-[11px] text-amber-300/90 pt-1">{t("settings.webStreamTosNotice")}</p>
                )}
                <div className="flex flex-wrap items-center gap-1.5 pt-1.5">
                  <span className="text-[10px] text-[#7C8596] font-mono">Presets:</span>
                  <button
                    type="button"
                    onClick={() =>
                      setCfg({
                        ...cfg,
                        camera: {
                          ...cfg.camera,
                          name: "Android IP Webcam",
                          source_type: "http_snapshot",
                          source_url: "http://192.168.1.50:8080/shot.jpg",
                          capture_interval_seconds: 15,
                          fov_horizontal_deg: 72,
                        },
                      })
                    }
                    className="px-2 py-0.5 rounded bg-white/[0.05] hover:bg-white/[0.1] border border-white/[0.08] text-[10px] font-mono text-[#C5CBD5]"
                  >
                    Android IP Webcam
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      setCfg({
                        ...cfg,
                        camera: {
                          ...cfg.camera,
                          name: "ESP32-CAM Sky Node",
                          source_type: "http_snapshot",
                          source_url: "http://192.168.1.60/capture",
                          capture_interval_seconds: 20,
                          fov_horizontal_deg: 66,
                        },
                      })
                    }
                    className="px-2 py-0.5 rounded bg-white/[0.05] hover:bg-white/[0.1] border border-white/[0.08] text-[10px] font-mono text-[#C5CBD5]"
                  >
                    ESP32-CAM
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      setCfg({
                        ...cfg,
                        camera: {
                          ...cfg.camera,
                          name: "Raspberry Pi Sky Cam",
                          source_type: "rtsp",
                          source_url: "rtsp://192.168.1.70:8554/cam",
                          capture_interval_seconds: 15,
                          fov_horizontal_deg: 120,
                        },
                      })
                    }
                    className="px-2 py-0.5 rounded bg-white/[0.05] hover:bg-white/[0.1] border border-white/[0.08] text-[10px] font-mono text-[#C5CBD5]"
                  >
                    Raspberry Pi Cam
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      setCfg({
                        ...cfg,
                        camera: {
                          ...cfg.camera,
                          name: "AllSky Local Folder",
                          source_type: "allsky_folder",
                          source_url: "./data/allsky/current",
                          capture_interval_seconds: 30,
                          fov_horizontal_deg: 165,
                        },
                      })
                    }
                    className="px-2 py-0.5 rounded bg-white/[0.05] hover:bg-white/[0.1] border border-white/[0.08] text-[10px] font-mono text-[#C5CBD5]"
                  >
                    AllSky Folder
                  </button>
                </div>
              </label>
            )}

            {testResult && (
              <div
                className={`p-2.5 rounded-lg border ${
                  testResult.ok
                    ? "bg-emerald-950/35 border-emerald-500/35 text-emerald-300"
                    : "bg-rose-950/35 border-rose-500/35 text-rose-300"
                }`}
              >
                {testResult.message} ({testResult.latency_ms} ms)
              </div>
            )}

            <div className="grid grid-cols-3 gap-2">
              <label className="space-y-1">
                <span className="text-[#9AA1AD]">{t("settings.captureSec")}</span>
                <input
                  type="number"
                  min={1}
                  value={cfg.camera.capture_interval_seconds}
                  onChange={(e) =>
                    setCfg({ ...cfg, camera: { ...cfg.camera, capture_interval_seconds: Number(e.target.value) } })
                  }
                  className="obs-input w-full font-mono tabular-nums"
                />
              </label>
              <label className="space-y-1">
                <span className="text-[#9AA1AD]">{t("settings.analysisSec")}</span>
                <input
                  type="number"
                  min={1}
                  value={cfg.camera.analysis_interval_seconds}
                  onChange={(e) =>
                    setCfg({ ...cfg, camera: { ...cfg.camera, analysis_interval_seconds: Number(e.target.value) } })
                  }
                  className="obs-input w-full font-mono tabular-nums"
                />
              </label>
              <label className="space-y-1">
                <span className="text-[#9AA1AD]">{t("settings.fov")}</span>
                <input
                  type="number"
                  min={10}
                  max={180}
                  value={cfg.camera.fov_horizontal_deg ?? 72}
                  onChange={(e) =>
                    setCfg({ ...cfg, camera: { ...cfg.camera, fov_horizontal_deg: Number(e.target.value) } })
                  }
                  className="obs-input w-full font-mono tabular-nums"
                />
              </label>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <label className="space-y-1">
                <span className="text-[#9AA1AD]">{t("settings.locationOpt")}</span>
                <input
                  type="text"
                  placeholder={t("settings.locationPh")}
                  value={cfg.camera.location_description}
                  onChange={(e) =>
                    setCfg({ ...cfg, camera: { ...cfg.camera, location_description: e.target.value } })
                  }
                  className="obs-input w-full"
                />
              </label>
              <label className="space-y-1">
                <span className="text-[#9AA1AD]">{t("settings.azEl")}</span>
                <div className="flex gap-1.5">
                  <input
                    type="number"
                    value={cfg.camera.orientation_azimuth_deg}
                    onChange={(e) =>
                      setCfg({ ...cfg, camera: { ...cfg.camera, orientation_azimuth_deg: Number(e.target.value) } })
                    }
                    className="obs-input w-1/2 font-mono tabular-nums"
                  />
                  <input
                    type="number"
                    value={cfg.camera.orientation_elevation_deg}
                    onChange={(e) =>
                      setCfg({ ...cfg, camera: { ...cfg.camera, orientation_elevation_deg: Number(e.target.value) } })
                    }
                    className="obs-input w-1/2 font-mono tabular-nums"
                  />
                </div>
              </label>
            </div>
          </div>

          <div className="obs-panel p-5 space-y-3">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Shield className="w-4 h-4 text-emerald-400" /> {t("settings.aiTitle")}
            </h3>
            <div className="grid grid-cols-3 gap-2">
              <label className="space-y-1">
                <span className="text-[#9AA1AD]">{t("settings.device")}</span>
                <select
                  value={cfg.ai.device}
                  onChange={(e) => setCfg({ ...cfg, ai: { ...cfg.ai, device: e.target.value } })}
                  className="obs-input w-full"
                >
                  <option value="auto">{t("settings.dev.auto")}</option>
                  <option value="cuda">{t("hw.dev.cuda")}</option>
                  <option value="rocm">{t("hw.dev.rocm")}</option>
                  <option value="directml">{t("hw.dev.directml")}</option>
                  <option value="mps">{t("hw.dev.mps")}</option>
                  <option value="cpu">{t("settings.dev.cpu")}</option>
                </select>
              </label>
              <label className="space-y-1">
                <span className="text-[#9AA1AD]">{t("settings.infRes")}</span>
                <select
                  value={cfg.vision.inference_resolution}
                  onChange={(e) =>
                    setCfg({ ...cfg, vision: { ...cfg.vision, inference_resolution: Number(e.target.value) } })
                  }
                  className="obs-input w-full font-mono tabular-nums"
                >
                  <option value={256}>256×256</option>
                  <option value={384}>384×384</option>
                  <option value={512}>512×512</option>
                  <option value={640}>640×640</option>
                </select>
              </label>
              <label className="space-y-1">
                <span className="text-[#9AA1AD]">{t("settings.vramCap")}</span>
                <input
                  type="number"
                  step="0.05"
                  min={0.2}
                  max={0.95}
                  value={cfg.ai.max_gpu_memory_fraction}
                  onChange={(e) =>
                    setCfg({ ...cfg, ai: { ...cfg.ai, max_gpu_memory_fraction: Number(e.target.value) } })
                  }
                  className="obs-input w-full font-mono tabular-nums"
                />
              </label>
            </div>

            <div className="grid grid-cols-3 gap-2">
              <label className="space-y-1">
                <span className="text-[#9AA1AD]">{t("settings.rawRet")}</span>
                <input
                  type="number"
                  min={1}
                  value={cfg.storage.raw_retention_days}
                  onChange={(e) =>
                    setCfg({ ...cfg, storage: { ...cfg.storage, raw_retention_days: Number(e.target.value) } })
                  }
                  className="obs-input w-full font-mono tabular-nums"
                />
              </label>
              <label className="space-y-1">
                <span className="text-[#9AA1AD]">{t("settings.procRet")}</span>
                <input
                  type="number"
                  min={1}
                  value={cfg.storage.processed_retention_days}
                  onChange={(e) =>
                    setCfg({ ...cfg, storage: { ...cfg.storage, processed_retention_days: Number(e.target.value) } })
                  }
                  className="obs-input w-full font-mono tabular-nums"
                />
              </label>
              <label className="space-y-1">
                <span className="text-[#9AA1AD]">{t("settings.evRet")}</span>
                <input
                  type="number"
                  min={1}
                  value={cfg.storage.event_retention_days}
                  onChange={(e) =>
                    setCfg({ ...cfg, storage: { ...cfg.storage, event_retention_days: Number(e.target.value) } })
                  }
                  className="obs-input w-full font-mono tabular-nums"
                />
              </label>
            </div>

            {system && (
              <div className="obs-inset p-3.5 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-white flex items-center gap-1.5">
                    <HardDrive className="w-3.5 h-3.5 text-sky-400" /> {t("settings.telemetryTitle")}
                  </span>
                  <button
                    type="button"
                    onClick={() => api.cleanupStorage().then(onConfigSaved)}
                    className="obs-btn-secondary px-2.5 py-0.5 text-[11px]"
                  >
                    {t("settings.runCleanup")}
                  </button>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono tabular-nums text-[11px] text-[#D2D6DF]">
                  <div>CPU: {system.hardware.cpu_percent}%</div>
                  <div>RAM: {system.hardware.ram_used_gb}/{system.hardware.ram_total_gb} GB</div>
                  <div>DB: {system.storage.database_size_mb} MB</div>
                  <div>Img: {(system.storage.raw_images_mb + system.storage.processed_images_mb).toFixed(1)} MB</div>
                </div>
                <div className="text-[11px] text-emerald-400">
                  Offline: {system.offline_mode ? t("settings.offlineActive") : t("settings.offlineCfg")} · GPU:{" "}
                  {system.hardware.gpu.device_name}
                </div>
              </div>
            )}
          </div>
        </div>

        <CameraTiltCalibrator
          value={{
            orientation_elevation_deg: Number(cfg.camera.orientation_elevation_deg ?? 65),
            orientation_azimuth_deg: Number(cfg.camera.orientation_azimuth_deg ?? 0),
            orientation_roll_deg: Number(cfg.camera.orientation_roll_deg ?? 0),
            fov_horizontal_deg: Number(cfg.camera.fov_horizontal_deg ?? 72),
            fov_vertical_deg: Number(cfg.camera.fov_vertical_deg ?? 45),
            assumed_cloud_altitude_m: cfg.camera.assumed_cloud_altitude_m ?? null,
            tilt_correction_enabled: Boolean(cfg.camera.tilt_correction_enabled ?? true),
          }}
          onChange={(nextGeom: CameraGeometryValues) =>
            setCfg({ ...cfg, camera: { ...cfg.camera, ...nextGeom } })
          }
          onApply={handleApplyGeometry}
          applying={applyingGeom}
          solar={(system?.camera as any)?.geometry?.solar || null}
          effectiveCloudAltitudeM={Number((system?.camera as any)?.geometry?.cloud_altitude_m || 2000)}
        />
      </form>

      <div className="obs-panel p-5 space-y-4 text-xs">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-semibold text-white">{t("settings.maskTitle")}</h3>
            <p className="text-[#8E95A2]">{t("settings.maskDesc")}</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <input
              type="text"
              value={polyLabel}
              onChange={(e) => setPolyLabel(e.target.value)}
              placeholder={t("settings.maskLabelPh")}
              className="obs-input w-28"
            />
            <select
              value={polyCategory}
              onChange={(e) => setPolyCategory(e.target.value as any)}
              className="obs-input"
            >
              <option value="trees">{tEnum("trees")}</option>
              <option value="buildings">{tEnum("buildings")}</option>
              <option value="poles">{tEnum("poles")}</option>
              <option value="roofs">{tEnum("roofs")}</option>
              <option value="road">{tEnum("road")}</option>
              <option value="logo">{tEnum("logo")}</option>
              <option value="other">{tEnum("other")}</option>
            </select>
            <button
              type="button"
              disabled={newPolyPts.length < 3}
              onClick={addExclusionPolygon}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white disabled:opacity-40"
            >
              <Plus className="w-3.5 h-3.5" /> {t("settings.addExclusion", { n: newPolyPts.length })}
            </button>
            <button
              type="button"
              disabled={newPolyPts.length < 3}
              onClick={setAsSkyRoi}
              className="obs-btn-primary px-2.5 py-1.5"
            >
              {t("settings.setSkyRoi")}
            </button>
            <button
              type="button"
              onClick={() => setNewPolyPts([])}
              className="obs-btn-secondary px-2.5 py-1.5"
            >
              {t("settings.clearDraft")}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div
            onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const nx = Number(((e.clientX - rect.left) / rect.width).toFixed(3));
              const ny = Number(((e.clientY - rect.top) / rect.height).toFixed(3));
              setNewPolyPts((pts) => [...pts, [nx, ny]]);
            }}
            className="lg:col-span-2 relative aspect-video bg-[#08090B] border border-white/[0.08] rounded-xl overflow-hidden cursor-crosshair"
          >
            <img
              src={`/api/camera/latest?layer=mask&t=${Date.now()}`}
              onError={(e) => {
                (e.currentTarget as HTMLImageElement).style.display = "none";
              }}
              alt="Mask preview"
              className="w-full h-full object-contain pointer-events-none"
            />
            <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 w-full h-full pointer-events-none">
              {newPolyPts.length >= 2 && (
                <polygon
                  points={newPolyPts.map((p) => `${p[0] * 100},${p[1] * 100}`).join(" ")}
                  fill="rgba(56, 189, 248, 0.2)"
                  stroke="#38BDF8"
                  strokeWidth="0.6"
                />
              )}
              {newPolyPts.map((p, i) => (
                <circle key={i} cx={p[0] * 100} cy={p[1] * 100} r="1.2" fill="#38BDF8" />
              ))}
            </svg>
          </div>

          <div className="space-y-2">
            <div className="obs-label">{t("settings.savedExclusions")}</div>
            {(cfg.camera.exclusion_polygons || []).map((p: any) => (
              <div key={p.id} className="obs-inset flex items-center justify-between p-2.5">
                <span>
                  {p.label}{" "}
                  <span className="text-[#8E95A2]">
                    ({tEnum(p.category)}, {t("settings.pts", { n: p.points.length })})
                  </span>
                </span>
                <button
                  type="button"
                  onClick={() => removeExclusionPolygon(p.id)}
                  className="text-rose-400 hover:text-rose-300"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
            {(cfg.camera.exclusion_polygons || []).length === 0 && (
              <p className="text-[#646B78]">{t("settings.noExclusions")}</p>
            )}
          </div>
        </div>

        <div className="pt-4 border-t border-white/[0.07]">
          <ExclusionMaskPainter
            imageUrl={`/api/camera/latest?layer=raw&t=${Date.now()}`}
            onSaved={onConfigSaved}
          />
        </div>
      </div>

      {/* Data, Image Storage & System Reset Center */}
      <div className="obs-panel p-5 space-y-4 text-xs">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Database className="w-4 h-4 text-rose-400" /> {t("data.title")}
            </h3>
            <p className="text-[#8E95A2] mt-0.5">{t("data.subtitle")}</p>
          </div>
          {storageInfo && (
            <span className="px-3 py-1 rounded-lg bg-[#0F1116] border border-white/[0.08] font-mono text-[11px] text-sky-300">
              {t("data.totalUsed", {
                mb: Number(storageInfo.total_app_storage_mb ?? 0).toFixed(1),
                free: Number(storageInfo.disk_free_gb ?? 0).toFixed(1),
              })}
            </span>
          )}
        </div>

        {purgeMsg && (
          <div className="px-4 py-2.5 rounded-lg bg-emerald-950/40 border border-emerald-500/35 text-emerald-300 flex items-center gap-2">
            <CheckCircle className="w-4 h-4 shrink-0" />
            <span>{purgeMsg}</span>
          </div>
        )}

        {/* Live Storage Breakdown Cards */}
        {storageInfo && (
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2 font-mono text-[11px]">
            <div className="obs-inset p-2.5">
              <div className="text-[#8E95A2] text-[10px]">Foto Raw</div>
              <div className="text-white font-bold mt-0.5">{storageInfo.raw_images_mb} MB</div>
              <div className="text-[#646B78] text-[10px]">
                {t("data.filesUnit", { n: storageInfo.file_counts?.raw_images ?? 0 })}
              </div>
            </div>
            <div className="obs-inset p-2.5">
              <div className="text-[#8E95A2] text-[10px]">Segment./Moto</div>
              <div className="text-white font-bold mt-0.5">{storageInfo.processed_images_mb} MB</div>
              <div className="text-[#646B78] text-[10px]">
                {t("data.filesUnit", { n: storageInfo.file_counts?.processed_images ?? 0 })}
              </div>
            </div>
            <div className="obs-inset p-2.5">
              <div className="text-[#8E95A2] text-[10px]">Osservazioni DB</div>
              <div className="text-sky-300 font-bold mt-0.5">
                {storageInfo.db_counts?.observations ?? 0}
              </div>
              <div className="text-[#646B78] text-[10px]">DB {storageInfo.database_size_mb} MB</div>
            </div>
            <div className="obs-inset p-2.5">
              <div className="text-[#8E95A2] text-[10px]">Previsioni</div>
              <div className="text-white font-bold mt-0.5">
                {storageInfo.db_counts?.forecasts ?? 0}
              </div>
              <div className="text-[#646B78] text-[10px]">
                Verif: {storageInfo.db_counts?.verifications ?? 0}
              </div>
            </div>
            <div className="obs-inset p-2.5">
              <div className="text-[#8E95A2] text-[10px]">Eventi / Avvisi</div>
              <div className="text-amber-300 font-bold mt-0.5">
                {storageInfo.db_counts?.events ?? 0}
              </div>
              <div className="text-[#646B78] text-[10px]">{storageInfo.event_images_mb} MB</div>
            </div>
            <div className="obs-inset p-2.5">
              <div className="text-[#8E95A2] text-[10px]">Timelapse</div>
              <div className="text-white font-bold mt-0.5">{storageInfo.timelapse_mb} MB</div>
              <div className="text-[#646B78] text-[10px]">
                {t("data.filesUnit", { n: storageInfo.file_counts?.timelapse ?? 0 })}
              </div>
            </div>
            <div className="obs-inset p-2.5">
              <div className="text-[#8E95A2] text-[10px]">Dataset & AI</div>
              <div className="text-white font-bold mt-0.5">
                {(Number(storageInfo.datasets_mb || 0) + Number(storageInfo.models_mb || 0)).toFixed(1)} MB
              </div>
              <div className="text-[#646B78] text-[10px]">
                {storageInfo.db_counts?.dataset_items ?? 0} item · {storageInfo.db_counts?.models ?? 0} mod
              </div>
            </div>
            <div className="obs-inset p-2.5">
              <div className="text-[#8E95A2] text-[10px]">Mappe Offline</div>
              <div className="text-emerald-300 font-bold mt-0.5">{storageInfo.map_tiles_mb ?? 0} MB</div>
              <div className="text-[#646B78] text-[10px]">
                {t("data.filesUnit", { n: storageInfo.file_counts?.map_tiles ?? 0 })}
              </div>
            </div>
          </div>
        )}

        {/* 1-Click Preset Action Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
          <button
            type="button"
            onClick={() =>
              setPendingPurgeTargets([
                "raw_images",
                "processed_images",
                "observations",
                "forecasts",
                "events",
                "timelapse",
              ])
            }
            className="obs-inset p-3.5 text-left hover:border-sky-400/40 border border-white/[0.07] transition space-y-1"
          >
            <div className="font-semibold text-sky-300 flex items-center gap-1.5">
              <RotateCcw className="w-3.5 h-3.5" /> {t("data.preset.restartMonitoring")}
            </div>
            <p className="text-[11px] text-[#8E95A2]">{t("data.preset.restartMonitoringDesc")}</p>
          </button>

          <button
            type="button"
            onClick={() => setPendingPurgeTargets(["raw_images", "processed_images"])}
            className="obs-inset p-3.5 text-left hover:border-indigo-400/40 border border-white/[0.07] transition space-y-1"
          >
            <div className="font-semibold text-indigo-300 flex items-center gap-1.5">
              <Trash2 className="w-3.5 h-3.5" /> {t("data.preset.imagesOnly")}
            </div>
            <p className="text-[11px] text-[#8E95A2]">{t("data.preset.imagesOnlyDesc")}</p>
          </button>

          <button
            type="button"
            onClick={() => setPendingPurgeTargets(["forecasts", "events", "timelapse"])}
            className="obs-inset p-3.5 text-left hover:border-amber-400/40 border border-white/[0.07] transition space-y-1"
          >
            <div className="font-semibold text-amber-300 flex items-center gap-1.5">
              <Trash2 className="w-3.5 h-3.5" /> {t("data.preset.clearLogs")}
            </div>
            <p className="text-[11px] text-[#8E95A2]">{t("data.preset.clearLogsDesc")}</p>
          </button>

          <button
            type="button"
            onClick={() =>
              setPendingPurgeTargets([
                "raw_images",
                "processed_images",
                "observations",
                "forecasts",
                "events",
                "timelapse",
                "exclusion_masks",
                "datasets",
                "models",
                "reset_config",
              ])
            }
            className="obs-inset p-3.5 text-left hover:border-rose-500/60 border border-rose-500/25 bg-rose-950/15 transition space-y-1"
          >
            <div className="font-semibold text-rose-300 flex items-center gap-1.5">
              <AlertTriangle className="w-3.5 h-3.5" /> {t("data.preset.factoryReset")}
            </div>
            <p className="text-[11px] text-rose-200/75">{t("data.preset.factoryResetDesc")}</p>
          </button>
        </div>

        {/* Confirmation Banner when a Preset or Custom Delete is triggered */}
        {pendingPurgeTargets && pendingPurgeTargets.length > 0 && (
          <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-500/40 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="font-semibold text-rose-200 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-rose-400" />
                {t("data.confirmTitle")} ({pendingPurgeTargets.length})
              </div>
              <div className="flex flex-wrap gap-1.5">
                {pendingPurgeTargets.map((tid) => (
                  <span
                    key={tid}
                    className="px-2 py-0.5 rounded bg-rose-500/20 border border-rose-400/30 text-[11px] font-mono text-rose-200"
                  >
                    {t(`data.cat.${tid}`)}
                  </span>
                ))}
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-4 text-[11px] text-[#D2D6DF]">
              {!pendingPurgeTargets.includes("datasets") && (
                <label className="inline-flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={preserveTraining}
                    onChange={(e) => setPreserveTraining(e.target.checked)}
                  />
                  <span>{t("data.preserveTrainOpt")}</span>
                </label>
              )}
              {pendingPurgeTargets.includes("reset_config") && (
                <label className="inline-flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={preserveCam}
                    onChange={(e) => setPreserveCam(e.target.checked)}
                  />
                  <span>{t("data.preserveCamOpt")}</span>
                </label>
              )}
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                disabled={purging}
                onClick={() => executePurge(pendingPurgeTargets)}
                className="px-4 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-semibold disabled:opacity-50"
              >
                {t("data.confirmExecute")}
              </button>
              <button
                type="button"
                disabled={purging}
                onClick={() => setPendingPurgeTargets(null)}
                className="obs-btn-secondary px-3 py-1.5"
              >
                {t("data.cancel")}
              </button>
            </div>
          </div>
        )}

        {/* Custom Selective Purge Checkboxes */}
        <div className="obs-inset p-3.5 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="font-semibold text-white">{t("data.customTitle")}</span>
            <button
              type="button"
              disabled={selectedTargets.length === 0 || purging}
              onClick={() => setPendingPurgeTargets(selectedTargets)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 border border-rose-500/35 text-rose-200 font-semibold disabled:opacity-40 transition"
            >
              <Trash2 className="w-3.5 h-3.5" />
              {t("data.deleteSelectedBtn", { n: selectedTargets.length })}
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {PURGE_CATEGORIES.map((cat) => {
              const checked = selectedTargets.includes(cat.id);
              return (
                <label
                  key={cat.id}
                  className={`flex items-center gap-2.5 p-2 rounded-lg border cursor-pointer transition ${
                    checked
                      ? "bg-rose-500/10 border-rose-400/35 text-white"
                      : "bg-[#0B0C0E] border-white/[0.06] text-[#9AA1AD] hover:text-white"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => togglePurgeTarget(cat.id)}
                    className="accent-rose-500"
                  />
                  <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: cat.color }} />
                  <span className="truncate">{t(`data.cat.${cat.id}`)}</span>
                </label>
              );
            })}
          </div>
        </div>
      </div>

      {/* Multi-Camera 360° Sky Fusion (Secondary Cameras) */}
      <div className="obs-panel p-5 space-y-4 text-xs">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Wifi className="w-4 h-4 text-emerald-400" /> {t("multicam.title")}
            </h3>
            <p className="text-[#8E95A2] mt-0.5">{t("multicam.subtitle")}</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() =>
                setSecondaryCams((prev) => [
                  ...prev,
                  {
                    id: `cam_${prev.length + 2}`,
                    name: `Sky Cam ${prev.length + 2}`,
                    enabled: true,
                    source_type: "rtsp",
                    source_url: "",
                    device_index: prev.length + 1,
                    orientation_azimuth_deg: (Number(cfg?.camera?.orientation_azimuth_deg ?? 180) + 180) % 360,
                    orientation_elevation_deg: 55,
                    orientation_roll_deg: 0,
                    fov_horizontal_deg: 74,
                    fov_vertical_deg: 54,
                  },
                ])
              }
              className="obs-btn-secondary inline-flex items-center gap-1 px-3 py-1.5"
            >
              <Plus className="w-3.5 h-3.5 text-emerald-400" /> {t("multicam.addBtn")}
            </button>
            <button
              type="button"
              disabled={multiCamBusy}
              onClick={async () => {
                setMultiCamBusy(true);
                try {
                  const res = await api.updateSecondaryCameras(secondaryCams);
                  setSecondaryCams(res.cameras || []);
                  setSaveMsg(t("multicam.savedMsg"));
                  onConfigSaved();
                } catch (err: any) {
                  setSaveMsg(`Error: ${err.message}`);
                } finally {
                  setMultiCamBusy(false);
                }
              }}
              className="obs-btn-primary inline-flex items-center gap-1 px-3 py-1.5"
            >
              <Save className="w-3.5 h-3.5" /> {t("multicam.saveBtn")}
            </button>
          </div>
        </div>

        {secondaryCams.length === 0 ? (
          <p className="text-[#646B78]">{t("multicam.empty")}</p>
        ) : (
          <div className="space-y-2.5">
            {secondaryCams.map((sc, idx) => (
              <div
                key={sc.id || idx}
                className="obs-inset p-3 grid grid-cols-1 md:grid-cols-12 gap-2.5 items-end"
              >
                <label className="md:col-span-2 space-y-1">
                  <span className="text-[#8E95A2]">{t("multicam.name")}</span>
                  <input
                    type="text"
                    value={sc.name}
                    onChange={(e) =>
                      setSecondaryCams((prev) =>
                        prev.map((item, i) => (i === idx ? { ...item, name: e.target.value } : item))
                      )
                    }
                    className="obs-input w-full"
                  />
                </label>
                <label className="md:col-span-2 space-y-1">
                  <span className="text-[#8E95A2]">{t("multicam.source")}</span>
                  <select
                    value={sc.source_type}
                    onChange={(e) =>
                      setSecondaryCams((prev) =>
                        prev.map((item, i) => (i === idx ? { ...item, source_type: e.target.value } : item))
                      )
                    }
                    className="obs-input w-full"
                  >
                    <option value="rtsp">RTSP</option>
                    <option value="web_stream">Web Stream (HLS / SkylineWebcams)</option>
                    <option value="http_snapshot">HTTP Snapshot</option>
                    <option value="mjpeg_stream">MJPEG</option>
                    <option value="webcam_device">Webcam</option>
                    <option value="demo_simulation">Demo</option>
                  </select>
                </label>
                <label className="md:col-span-3 space-y-1">
                  <span className="text-[#8E95A2]">{t("multicam.url")}</span>
                  <input
                    type="text"
                    value={sc.source_url || ""}
                    placeholder="rtsp://user:pass@192.168.1.x:554/stream1"
                    onChange={(e) =>
                      setSecondaryCams((prev) =>
                        prev.map((item, i) => (i === idx ? { ...item, source_url: e.target.value } : item))
                      )
                    }
                    className="obs-input w-full font-mono"
                  />
                </label>
                <label className="md:col-span-1 space-y-1">
                  <span className="text-[#8E95A2]">{t("multicam.azimuth")}</span>
                  <input
                    type="number"
                    min={0}
                    max={360}
                    value={sc.orientation_azimuth_deg}
                    onChange={(e) =>
                      setSecondaryCams((prev) =>
                        prev.map((item, i) =>
                          i === idx ? { ...item, orientation_azimuth_deg: Number(e.target.value) } : item
                        )
                      )
                    }
                    className="obs-input w-full font-mono tabular-nums"
                  />
                </label>
                <label className="md:col-span-1 space-y-1">
                  <span className="text-[#8E95A2]">{t("multicam.elevation")}</span>
                  <input
                    type="number"
                    min={5}
                    max={90}
                    value={sc.orientation_elevation_deg}
                    onChange={(e) =>
                      setSecondaryCams((prev) =>
                        prev.map((item, i) =>
                          i === idx ? { ...item, orientation_elevation_deg: Number(e.target.value) } : item
                        )
                      )
                    }
                    className="obs-input w-full font-mono tabular-nums"
                  />
                </label>
                <div className="md:col-span-3 flex items-center gap-1.5">
                  <button
                    type="button"
                    disabled={multiCamBusy}
                    onClick={async () => {
                      setMultiCamBusy(true);
                      try {
                        await api.updateSecondaryCameras(secondaryCams);
                        const snap = await api.snapshotSecondaryCamera(sc.id);
                        setSaveMsg(`${snap.name}: ${snap.cloud_coverage_pct}% clouds fused on 3D Radar.`);
                      } catch (err: any) {
                        setSaveMsg(`Error: ${err.message}`);
                      } finally {
                        setMultiCamBusy(false);
                      }
                    }}
                    className="obs-btn-secondary px-2.5 py-1.5 text-emerald-300"
                  >
                    {t("multicam.testCapture")}
                  </button>
                  <button
                    type="button"
                    onClick={() => setSecondaryCams((prev) => prev.filter((_, i) => i !== idx))}
                    className="px-2.5 py-1.5 rounded-lg bg-rose-500/15 hover:bg-rose-500/25 border border-rose-500/30 text-rose-300"
                  >
                    {t("multicam.remove")}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
