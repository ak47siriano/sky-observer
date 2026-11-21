import React, { useCallback, useEffect, useState } from "react";
import { Download, Film, Pause, Play, RefreshCw, SkipBack, Trash2 } from "lucide-react";
import { useI18n } from "../i18n/I18nContext";
import { api } from "../services/api";

export const TimelapsePage: React.FC = () => {
  const { t, tEnum } = useI18n();
  const [frames, setFrames] = useState<any[]>([]);
  const [digests, setDigests] = useState<any[]>([]);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [fps, setFps] = useState(4);
  const [layer, setLayer] = useState<"raw" | "segmentation" | "motion">("raw");
  const [dateFilter, setDateFilter] = useState("");
  const [includeHud, setIncludeHud] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [digestBusy, setDigestBusy] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const [deleteFiles, setDeleteFiles] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");

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
      api
        .getTimelapseDigests()
        .then((d) => setDigests(d.digests || []))
        .catch(() => {});
    },
    [dateFilter]
  );

  useEffect(() => {
    loadFrames(false);
  }, [loadFrames]);

  useEffect(() => {
    const timer = window.setInterval(() => loadFrames(true), 5000);
    return () => window.clearInterval(timer);
  }, [loadFrames]);

  useEffect(() => {
    if (!playing || frames.length <= 1) return;
    const id = window.setInterval(() => {
      setIndex((prev) => (prev + 1) % frames.length);
    }, Math.max(50, Math.round(1000 / fps)));
    return () => window.clearInterval(id);
  }, [playing, fps, frames.length]);

  const current = frames[index];

  const handleExport = async () => {
    setExporting(true);
    try {
      const res = await fetch(api.exportTimelapseUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          layer,
          fps: Math.max(4, fps),
          date: dateFilter || null,
          include_hud: includeHud,
        }),
      });
      if (!res.ok) throw new Error("Failed to export timelapse");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `skyobserver_timelapse_${layer}.mp4`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setExporting(false);
    }
  };

  const handleGenerateDigest = async () => {
    setDigestBusy(true);
    try {
      const res = await api.generateDailyDigest({
        date: dateFilter || null,
        layer,
        fps: Math.max(6, fps),
        include_hud: includeHud,
      });
      if (res.digests) setDigests(res.digests);
      if (res.digest) {
        setStatusMsg(t("timelapse.digestGenerated", { file: res.digest.filename, mb: res.digest.size_mb }));
        window.setTimeout(() => setStatusMsg(""), 5000);
      }
    } catch (err: any) {
      setStatusMsg(`Error: ${err.message}`);
    } finally {
      setDigestBusy(false);
    }
  };

  const handleClear = async () => {
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
      setStatusMsg(t("timelapse.clearedMsg", { n: res.cleared_frames ?? 0 }));
      window.setTimeout(() => setStatusMsg(""), 4000);
    } finally {
      setClearing(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-white">{t("timelapse.title")}</h2>
          <p className="text-xs text-slate-400">{t("timelapse.available", { n: frames.length })}</p>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs">
          <input
            type="date"
            value={dateFilter}
            onChange={(e) => setDateFilter(e.target.value)}
            className="bg-[#0F1623] border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200"
          />
          <select
            value={layer}
            onChange={(e) => setLayer(e.target.value as any)}
            className="bg-[#0F1623] border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200"
          >
            <option value="raw">{t("timelapse.layer.raw")}</option>
            <option value="segmentation">{t("timelapse.layer.seg")}</option>
            <option value="motion">{t("timelapse.layer.motion")}</option>
          </select>
          <button
            type="button"
            onClick={() => loadFrames(true)}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            {t("timelapse.refresh")}
          </button>
          <label className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-[#0F1623] border border-slate-700 text-slate-200 cursor-pointer">
            <input
              type="checkbox"
              checked={includeHud}
              onChange={(e) => setIncludeHud(e.target.checked)}
            />
            <span>{t("timelapse.hudToggle")}</span>
          </label>
          <button
            type="button"
            onClick={handleGenerateDigest}
            disabled={digestBusy || frames.length < 2}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/35 border border-emerald-500/40 text-emerald-200 font-medium disabled:opacity-40"
          >
            <Film className="w-3.5 h-3.5" />
            {t("timelapse.generateDigest")}
          </button>
          <button
            type="button"
            onClick={handleExport}
            disabled={exporting || frames.length < 2}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 text-white font-medium disabled:opacity-50"
          >
            <Download className="w-3.5 h-3.5" />
            {exporting ? t("timelapse.exporting") : t("timelapse.exportMp4")}
          </button>
          <button
            type="button"
            onClick={() => setConfirmClear((v) => !v)}
            disabled={clearing || frames.length === 0}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-600/20 hover:bg-rose-600/35 border border-rose-500/40 text-rose-200 font-medium disabled:opacity-40"
          >
            <Trash2 className="w-3.5 h-3.5" />
            {clearing ? t("timelapse.clearing") : t("timelapse.clearBtn")}
          </button>
        </div>
      </div>

      {confirmClear && (
        <div className="bg-rose-950/30 border border-rose-700/60 rounded-xl p-3.5 flex flex-wrap items-center justify-between gap-3 text-xs">
          <div className="space-y-1.5">
            <p className="font-semibold text-rose-200">
              {dateFilter
                ? t("timelapse.confirmTitleDate", { n: frames.length, date: dateFilter })
                : t("timelapse.confirmTitleAll", { n: frames.length })}
            </p>
            <label className="flex items-center gap-2 text-slate-300 cursor-pointer">
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
              onClick={handleClear}
              className="px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-medium disabled:opacity-50"
            >
              {t("timelapse.confirmBtn")}
            </button>
            <button
              type="button"
              disabled={clearing}
              onClick={() => setConfirmClear(false)}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200"
            >
              {t("timelapse.cancelBtn")}
            </button>
          </div>
        </div>
      )}

      {statusMsg && (
        <div className="bg-emerald-950/30 border border-emerald-700/50 rounded-xl px-4 py-2.5 text-xs text-emerald-300">
          {statusMsg}
        </div>
      )}

      <div className="bg-[#0F1623] border border-slate-800 rounded-xl overflow-hidden">
        <div className="aspect-video bg-[#060911] flex items-center justify-center relative">
          {current ? (
            <img
              src={`/api/analysis/image/${current.observation_id}?layer=${layer}`}
              alt="Timelapse frame"
              className="w-full h-full object-contain"
            />
          ) : (
            <div className="text-center space-y-1 px-4">
              <p className="text-sm text-slate-400">{t("timelapse.empty")}</p>
              <p className="text-xs text-slate-500">{t("timelapse.emptyHint")}</p>
            </div>
          )}
          {current && (
            <div className="absolute bottom-3 left-3 bg-slate-950/80 border border-slate-800 rounded px-3 py-1 text-xs font-mono text-slate-200">
              {new Date(current.captured_at).toLocaleString()} · {t("overview.cloudCoverage")}{" "}
              {current.cloud_coverage_pct.toFixed(1)}% · {tEnum(current.visual_condition)}
            </div>
          )}
        </div>

        <div className="p-4 border-t border-slate-800 flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                setPlaying(false);
                setIndex(0);
              }}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200"
            >
              <SkipBack className="w-4 h-4" />
            </button>
            <button
              onClick={() => setPlaying((p) => !p)}
              disabled={frames.length === 0}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-xs font-medium disabled:opacity-50"
            >
              {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {playing ? t("timelapse.pause") : t("timelapse.play")}
            </button>
          </div>

          <input
            type="range"
            min={0}
            max={Math.max(0, frames.length - 1)}
            value={index}
            onChange={(e) => setIndex(Number(e.target.value))}
            className="flex-1 accent-sky-500"
          />

          <div className="flex items-center gap-2 text-xs text-slate-300">
            <span>{t("timelapse.speed", { fps })}</span>
            <input
              type="range"
              min={1}
              max={24}
              value={fps}
              onChange={(e) => setFps(Number(e.target.value))}
              className="w-24 accent-sky-500"
            />
          </div>
        </div>
      </div>

      {/* Saved Daily Sky Digests */}
      <div className="bg-[#0F1623] border border-slate-800 rounded-xl p-4 space-y-2.5 text-xs">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <Film className="w-4 h-4 text-emerald-400" /> {t("timelapse.digestsTitle")}
        </h3>
        {digests.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {digests.map((dg) => (
              <div
                key={dg.filename}
                className="flex items-center justify-between gap-2 p-2.5 rounded-lg bg-slate-900/70 border border-slate-800"
              >
                <div className="min-w-0">
                  <div className="font-mono font-semibold text-white truncate">{dg.filename}</div>
                  <div className="text-[11px] text-slate-400 font-mono">
                    {dg.date} · {dg.size_mb} MB
                  </div>
                </div>
                <a
                  href={dg.download_url}
                  download={dg.filename}
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-sky-600/20 hover:bg-sky-600/35 border border-sky-500/35 text-sky-200 font-medium shrink-0"
                >
                  <Download className="w-3 h-3" /> MP4
                </a>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-slate-400">{t("timelapse.noDigests")}</p>
        )}
      </div>
    </div>
  );
};
