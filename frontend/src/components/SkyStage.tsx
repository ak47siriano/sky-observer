import React, { useCallback, useEffect, useState } from "react";
import {
  Brush,
  Camera,
  Columns,
  Compass,
  Download,
  Film,
  Grid3X3,
  Image as ImageIcon,
  Layers,
  Pause,
  Play,
  RefreshCw,
  SkipBack,
  Trash2,
  Upload,
  Wind,
} from "lucide-react";
import { useI18n } from "../i18n/I18nContext";
import { api } from "../services/api";
import type { CameraStatus, Observation } from "../types/api";
import { ExclusionMaskPainter } from "./ExclusionMaskPainter";
import { SkyHorizonOverlay } from "./SkyHorizonOverlay";

type StageMode = "raw" | "segmentation" | "compare" | "motion" | "timelapse" | "exclude";

export interface SkyStageProps {
  camera: CameraStatus | null;
  obs: Observation | null;
  onTriggerSnapshot: () => void;
  onUploadFile: (file: File) => void;
  onMaskSaved: () => void;
  busy: boolean;
}

const SEG_SWATCHES = [
  { key: "vision.legend.clear", color: "#2563EB", field: "clear_sky_pct" },
  { key: "vision.legend.thin", color: "#38BDF8", field: "thin_cloud_pct" },
  { key: "vision.legend.dense", color: "#F8FAFC", field: "dense_cloud_pct" },
  { key: "vision.legend.dark", color: "#C34BEB", field: "dark_cloud_pct" },
  { key: "vision.legend.sunGlare", color: "#FBBF24", field: "sun_glare_pct" },
] as const;

export const SkyStage: React.FC<SkyStageProps> = ({
  camera,
  obs,
  onTriggerSnapshot,
  onUploadFile,
  onMaskSaved,
  busy,
}) => {
  const { t, tEnum, tCardinal } = useI18n();
  const [mode, setMode] = useState<StageMode>("raw");
  const [showGrid, setShowGrid] = useState(false);
  const [showHorizon, setShowHorizon] = useState(false);
  const [splitPct, setSplitPct] = useState(50);
  const [draggingSplit, setDraggingSplit] = useState(false);
  const [excludeSub, setExcludeSub] = useState<"paint" | "preview">("paint");

  // Embedded Timelapse state
  const [frames, setFrames] = useState<any[]>([]);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [fps, setFps] = useState(4);
  const [tlLayer, setTlLayer] = useState<"raw" | "segmentation" | "motion">("raw");
  const [dateFilter, setDateFilter] = useState("");
  const [exporting, setExporting] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const [deleteFiles, setDeleteFiles] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [tlStatusMsg, setTlStatusMsg] = useState("");

  const loadFrames = useCallback(
    (preservePosition = false) => {
      api
        .getTimelapseFrames(dateFilter || undefined, 400)
        .then((d) => {
          const next = d.frames || [];
          setFrames(next);
          if (preservePosition) {
            setIndex((prev) => Math.min(prev, Math.max(0, next.length - 1)));
          } else {
            setIndex(0);
          }
        })
        .catch(() => {});
    },
    [dateFilter]
  );

  useEffect(() => {
    if (mode !== "timelapse") return;
    loadFrames(true);
    const id = window.setInterval(() => loadFrames(true), 5000);
    return () => window.clearInterval(id);
  }, [mode, loadFrames]);

  useEffect(() => {
    if (mode !== "timelapse" || !playing || frames.length <= 1) return;
    const id = window.setInterval(() => {
      setIndex((prev) => (prev + 1) % frames.length);
    }, Math.max(50, Math.round(1000 / fps)));
    return () => window.clearInterval(id);
  }, [mode, playing, fps, frames.length]);

  const handleExport = async () => {
    setExporting(true);
    try {
      const res = await fetch(api.exportTimelapseUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ layer: tlLayer, fps: Math.max(4, fps), date: dateFilter || null }),
      });
      if (!res.ok) throw new Error("Failed to export timelapse");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `skyobserver_timelapse_${tlLayer}.mp4`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setExporting(false);
    }
  };

  const handleClearTimelapse = async () => {
    setClearing(true);
    try {
      const res = await api.clearTimelapse({
        date: dateFilter || null,
        delete_files: deleteFiles,
      });
      setPlaying(false);
      setFrames([]);
      setIndex(0);
      setConfirmClear(false);
      setDeleteFiles(false);
      setTlStatusMsg(t("timelapse.clearedMsg", { n: res.cleared_frames ?? 0 }));
      window.setTimeout(() => setTlStatusMsg(""), 4000);
    } finally {
      setClearing(false);
    }
  };

  const currentTl = frames[index];
  const tsQuery = obs ? encodeURIComponent(obs.captured_at) : String(Date.now());
  const rawImgUrl = obs ? `/api/analysis/image/${obs.id}?layer=raw&t=${tsQuery}` : `/api/camera/latest?layer=raw`;
  const segImgUrl = obs ? `/api/analysis/image/${obs.id}?layer=segmentation&t=${tsQuery}` : null;
  const activeLayerUrl = obs
    ? mode === "segmentation"
      ? segImgUrl
      : mode === "motion"
      ? `/api/analysis/image/${obs.id}?layer=motion&t=${tsQuery}`
      : rawImgUrl
    : null;
  const camGeom = (camera as any)?.geometry || null;

  const modes: Array<{ id: StageMode; label: string; icon: React.ComponentType<{ className?: string }> }> = [
    { id: "raw", label: t("stage.mode.raw"), icon: ImageIcon },
    { id: "segmentation", label: t("stage.mode.segmentation"), icon: Layers },
    { id: "compare", label: t("stage.mode.compare"), icon: Columns },
    { id: "motion", label: t("stage.mode.motion"), icon: Wind },
    { id: "timelapse", label: t("stage.mode.timelapse"), icon: Film },
    { id: "exclude", label: t("stage.mode.exclude"), icon: Brush },
  ];

  return (
    <div className="obs-panel overflow-hidden flex flex-col xl:h-full">
      {/* Compact Single-Row Instrument Toolbar */}
      <div className="px-3 py-2 bg-[#15181E] border-b border-[#262B34] flex flex-wrap items-center justify-between gap-2 shrink-0">
        <div className="inline-flex flex-wrap items-center gap-1">
          {modes.map(({ id, label, icon: Icon }) => {
            const active = mode === id;
            return (
              <button
                key={id}
                type="button"
                onClick={() => setMode(id)}
                className={active ? "obs-pill-active" : "obs-pill border-transparent bg-transparent"}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{label}</span>
              </button>
            );
          })}
        </div>

        {/* Context Controls */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {mode !== "timelapse" && mode !== "exclude" && (
            <button
              type="button"
              onClick={() => setShowHorizon((h) => !h)}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border transition ${
                showHorizon
                  ? "bg-sky-500/20 border-sky-400/50 text-sky-200"
                  : "obs-btn-secondary"
              }`}
            >
              <Compass className="w-3.5 h-3.5 text-sky-400" />
              {t("stage.horizonToggle")}
            </button>
          )}

          {(mode === "segmentation" || mode === "compare") && obs?.coverage?.grid?.length ? (
            <button
              type="button"
              onClick={() => setShowGrid((g) => !g)}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border transition ${
                showGrid
                  ? "bg-white/[0.12] border-white/25 text-white"
                  : "obs-btn-secondary"
              }`}
            >
              <Grid3X3 className="w-3.5 h-3.5" />
              {t("stage.gridToggle")}
            </button>
          ) : null}

          {mode === "exclude" && (
            <div className="inline-flex bg-[#0B0C0E] border border-white/[0.07] rounded-lg p-0.5">
              <button
                type="button"
                onClick={() => setExcludeSub("paint")}
                className={`px-2.5 py-1 rounded-md text-xs font-medium transition ${
                  excludeSub === "paint" ? "bg-[#F4F5F7] text-[#0B0C0E]" : "text-[#9AA1AD] hover:text-white"
                }`}
              >
                {t("stage.paintMask")}
              </button>
              <button
                type="button"
                onClick={() => setExcludeSub("preview")}
                className={`px-2.5 py-1 rounded-md text-xs font-medium transition ${
                  excludeSub === "preview" ? "bg-[#F4F5F7] text-[#0B0C0E]" : "text-[#9AA1AD] hover:text-white"
                }`}
              >
                {t("stage.maskPreview")}
              </button>
            </div>
          )}

          {mode !== "timelapse" ? (
            <>
              <button
                type="button"
                onClick={onTriggerSnapshot}
                disabled={busy}
                className="obs-btn-secondary inline-flex items-center gap-1.5 px-3 py-1.5"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${busy ? "animate-spin" : ""}`} />
                {t("overview.captureNow")}
              </button>
              <label className="obs-btn-secondary cursor-pointer inline-flex items-center gap-1.5 px-3 py-1.5">
                <Upload className="w-3.5 h-3.5" />
                {t("overview.upload")}
                <input
                  type="file"
                  accept="image/*,video/*"
                  className="hidden"
                  onChange={(e) => e.target.files?.[0] && onUploadFile(e.target.files[0])}
                />
              </label>
            </>
          ) : (
            <>
              <input
                type="date"
                value={dateFilter}
                onChange={(e) => setDateFilter(e.target.value)}
                className="obs-input text-xs py-1"
              />
              <select
                value={tlLayer}
                onChange={(e) => setTlLayer(e.target.value as any)}
                className="obs-input text-xs py-1"
              >
                <option value="raw">{t("timelapse.layer.raw")}</option>
                <option value="segmentation">{t("timelapse.layer.seg")}</option>
                <option value="motion">{t("timelapse.layer.motion")}</option>
              </select>
              <button
                type="button"
                onClick={() => loadFrames(true)}
                title={t("timelapse.refresh")}
                className="obs-btn-secondary inline-flex items-center gap-1 px-2.5 py-1.5"
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                onClick={handleExport}
                disabled={exporting || frames.length < 2}
                className="obs-btn-primary inline-flex items-center gap-1.5 px-3 py-1.5"
              >
                <Download className="w-3.5 h-3.5" />
                {exporting ? t("timelapse.exporting") : t("timelapse.exportMp4")}
              </button>
              <button
                type="button"
                onClick={() => setConfirmClear((v) => !v)}
                disabled={clearing || frames.length === 0}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-500/15 hover:bg-rose-500/25 border border-rose-500/35 text-rose-200 font-medium disabled:opacity-40 transition"
              >
                <Trash2 className="w-3.5 h-3.5" />
                {clearing ? t("timelapse.clearing") : t("timelapse.clearBtn")}
              </button>
            </>
          )}
        </div>
      </div>

      {/* Confirmation Banner for Clearing Timelapse */}
      {mode === "timelapse" && confirmClear && (
        <div className="px-4 py-3 bg-rose-950/35 border-b border-rose-500/30 flex flex-wrap items-center justify-between gap-3 text-xs">
          <div className="space-y-1.5">
            <p className="font-semibold text-rose-200">
              {dateFilter
                ? t("timelapse.confirmTitleDate", { n: frames.length, date: dateFilter })
                : t("timelapse.confirmTitleAll", { n: frames.length })}
            </p>
            <label className="flex items-center gap-2 text-[#D4D8DF] cursor-pointer">
              <input
                type="checkbox"
                checked={deleteFiles}
                onChange={(e) => setDeleteFiles(e.target.checked)}
              />
              <span>{t("timelapse.deleteFilesOpt")}</span>
            </label>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={clearing}
              onClick={handleClearTimelapse}
              className="px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-semibold disabled:opacity-50"
            >
              {t("timelapse.confirmBtn")}
            </button>
            <button
              type="button"
              disabled={clearing}
              onClick={() => setConfirmClear(false)}
              className="obs-btn-secondary px-3 py-1.5"
            >
              {t("timelapse.cancelBtn")}
            </button>
          </div>
        </div>
      )}

      {mode === "timelapse" && tlStatusMsg && (
        <div className="px-4 py-2 bg-emerald-950/35 border-b border-emerald-500/30 text-xs text-emerald-300">
          {tlStatusMsg}
        </div>
      )}

      {/* Stage Body */}
      {mode === "exclude" ? (
        excludeSub === "paint" ? (
          <div className="p-4">
            <ExclusionMaskPainter imageUrl={rawImgUrl} onSaved={onMaskSaved} />
          </div>
        ) : (
          <div className="aspect-video bg-[#08090B] flex items-center justify-center">
            <img
              src={`/api/camera/latest?layer=mask&t=${tsQuery}`}
              alt="Active sky ROI & exclusion mask"
              className="w-full h-full object-contain"
            />
          </div>
        )
      ) : mode === "timelapse" ? (
        <>
          <div className="aspect-video bg-[#08090B] flex items-center justify-center relative">
            {currentTl ? (
              <img
                src={`/api/analysis/image/${currentTl.observation_id}?layer=${tlLayer}`}
                alt="Timelapse frame"
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="text-center space-y-1.5 px-6">
                <Film className="w-8 h-8 text-[#4C5260] mx-auto" />
                <p className="text-sm text-[#9AA1AD]">{t("timelapse.empty")}</p>
                <p className="text-xs text-[#646B78]">{t("timelapse.emptyHint")}</p>
              </div>
            )}
          </div>

          <div className="px-4 py-3 border-t border-white/[0.07] bg-[#101217] flex flex-wrap items-center gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => {
                  setPlaying(false);
                  setIndex(0);
                }}
                className="obs-btn-secondary p-2"
              >
                <SkipBack className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                onClick={() => setPlaying((p) => !p)}
                disabled={frames.length === 0}
                className="obs-btn-primary inline-flex items-center gap-1.5 px-3 py-1.5"
              >
                {playing ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                {playing ? t("timelapse.pause") : t("timelapse.play")}
              </button>
            </div>

            <input
              type="range"
              min={0}
              max={Math.max(0, frames.length - 1)}
              value={index}
              onChange={(e) => setIndex(Number(e.target.value))}
              className="flex-1 accent-[#F4F5F7]"
            />

            <div className="flex items-center gap-3 font-mono tabular-nums text-[#9AA1AD]">
              {currentTl && (
                <span>
                  {new Date(currentTl.captured_at).toLocaleTimeString()} ·{" "}
                  {currentTl.cloud_coverage_pct.toFixed(0)}%
                </span>
              )}
              <span>
                {frames.length > 0 ? `${index + 1}/${frames.length}` : "0/0"}
              </span>
              <span className="flex items-center gap-1.5">
                {t("timelapse.speed", { fps })}
                <input
                  type="range"
                  min={1}
                  max={24}
                  value={fps}
                  onChange={(e) => setFps(Number(e.target.value))}
                  className="w-20 accent-[#F4F5F7]"
                />
              </span>
            </div>
          </div>
        </>
      ) : (
        <>
          <div
            onPointerDown={(e) => {
              if (mode !== "compare") return;
              e.currentTarget.setPointerCapture(e.pointerId);
              setDraggingSplit(true);
              const rect = e.currentTarget.getBoundingClientRect();
              setSplitPct(Math.max(2, Math.min(98, ((e.clientX - rect.left) / rect.width) * 100)));
            }}
            onPointerMove={(e) => {
              if (mode !== "compare" || !draggingSplit) return;
              const rect = e.currentTarget.getBoundingClientRect();
              setSplitPct(Math.max(2, Math.min(98, ((e.clientX - rect.left) / rect.width) * 100)));
            }}
            onPointerUp={(e) => {
              if (mode !== "compare") return;
              setDraggingSplit(false);
              if (e.currentTarget.hasPointerCapture(e.pointerId)) {
                e.currentTarget.releasePointerCapture(e.pointerId);
              }
            }}
            className={`flex-1 min-h-[340px] bg-[#06090E] flex items-center justify-center relative select-none ${
              mode === "compare" ? "cursor-ew-resize touch-none" : ""
            }`}
          >
            {mode === "compare" && obs && segImgUrl ? (
              <>
                <img src={rawImgUrl} alt="Sky raw" className="w-full h-full object-contain pointer-events-none" />
                <img
                  src={segImgUrl}
                  alt="Sky segmentation compare"
                  style={{ clipPath: `inset(0 ${100 - splitPct}% 0 0)` }}
                  className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                />
                <div
                  style={{ left: `${splitPct}%` }}
                  className="absolute top-0 bottom-0 w-0.5 bg-sky-400 shadow-[0_0_10px_#38BDF8] pointer-events-none"
                >
                  <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 w-7 h-7 rounded-full bg-[#0B0C0E]/90 border-2 border-sky-400 flex items-center justify-center text-sky-300 shadow-lg">
                    <Columns className="w-3.5 h-3.5" />
                  </div>
                </div>
                <span className="absolute top-3 left-3 px-2 py-0.5 rounded bg-black/70 border border-white/15 text-[10px] font-mono text-sky-300 pointer-events-none">
                  {t("stage.mode.segmentation")}
                </span>
                <span className="absolute top-3 right-3 px-2 py-0.5 rounded bg-black/70 border border-white/15 text-[10px] font-mono text-white pointer-events-none">
                  {t("stage.mode.raw")}
                </span>
              </>
            ) : activeLayerUrl ? (
              <img src={activeLayerUrl} alt={`Sky view ${mode}`} className="w-full h-full object-contain" />
            ) : (
              <div className="text-center space-y-2 p-8">
                <Camera className="w-9 h-9 text-[#4C5260] mx-auto" />
                <p className="text-sm text-[#9AA1AD]">{t("overview.noImageYet")}</p>
                <p className="text-xs text-[#646B78]">{t("overview.connectOrUpload")}</p>
              </div>
            )}

            {showHorizon && (
              <SkyHorizonOverlay
                elevationDeg={Number(camGeom?.orientation_elevation_deg ?? 65)}
                azimuthDeg={Number(camGeom?.orientation_azimuth_deg ?? 0)}
                rollDeg={Number(camGeom?.orientation_roll_deg ?? 0)}
                fovHorizontalDeg={Number(camGeom?.fov_horizontal_deg ?? 72)}
                fovVerticalDeg={Number(camGeom?.fov_vertical_deg ?? 45)}
                solar={camGeom?.solar || null}
              />
            )}

            {(mode === "segmentation" || mode === "compare") && showGrid && obs?.coverage?.grid && obs.coverage.grid.length > 0 && (
              <div className="absolute inset-0 grid grid-cols-3 grid-rows-3 pointer-events-none">
                {obs.coverage.grid.map((cell, idx) => (
                  <div
                    key={idx}
                    className="border border-white/15 flex items-center justify-center bg-black/15"
                  >
                    <span className="px-2 py-0.5 rounded bg-black/65 text-[11px] font-mono tabular-nums text-white">
                      {cell.cloud_pct != null ? `${cell.cloud_pct.toFixed(0)}%` : t("vision.masked")}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Slim Instrument Footer Strip (Zero Clutter) */}
          <div className="px-4 py-2.5 border-t border-white/[0.07] bg-[#101217] flex flex-wrap items-center justify-between gap-3 text-xs">
            {(mode === "segmentation" || mode === "compare") && obs ? (
              <div className="flex flex-wrap items-center gap-4 text-[#C5CBD5]">
                {SEG_SWATCHES.map((s) => (
                  <span key={s.key} className="inline-flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: s.color }} />
                    <span>{t(s.key)}</span>
                    <span className="font-mono tabular-nums text-[#8E95A2]">
                      {Number((obs as any)[s.field] ?? 0).toFixed(1)}%
                    </span>
                  </span>
                ))}
              </div>
            ) : mode === "motion" ? (
              <div className="font-mono tabular-nums text-[#C5CBD5]">
                {obs?.motion?.available
                  ? t("stage.motionSummary", {
                      speed: obs.motion_speed_px_s.toFixed(1),
                      dir: tCardinal(obs.motion_direction_cardinal),
                      deg: obs.motion_direction_deg.toFixed(0),
                      cons: (obs.motion_consistency * 100).toFixed(0),
                    })
                  : t("stage.noMotion")}
              </div>
            ) : (
              <div className="font-mono tabular-nums text-[#9AA1AD]">
                {obs
                  ? t("stage.frameMeta", {
                      id: obs.id,
                      w: obs.width,
                      h: obs.height,
                      ms: obs.inference_duration_ms.toFixed(0),
                    })
                  : t("overview.noFrame")}
              </div>
            )}

            <div className="flex items-center gap-3 font-mono tabular-nums text-[11px] text-[#8E95A2]">
              <span>{camera?.camera_name || "Sky Camera"}</span>
              <span>
                {obs ? new Date(obs.captured_at).toLocaleTimeString() : t("overview.never")}
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
