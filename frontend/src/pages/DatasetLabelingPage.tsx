import React, { useEffect, useRef, useState } from "react";
import { Brush, Eraser, Save } from "lucide-react";
import { useI18n } from "../i18n/I18nContext";
import { api } from "../services/api";
import type { Observation } from "../types/api";

const CLOUD_CLASSES = [
  "clear_sky",
  "cirrus",
  "cirrostratus",
  "cumulus",
  "stratocumulus",
  "stratus",
  "altocumulus",
  "altostratus",
  "cumulonimbus_like",
  "mixed_clouds",
  "unknown",
];

const EVENT_LABELS = [
  "normal",
  "rapid_cloud_development",
  "sky_clearing",
  "cloud_cover_increasing",
  "dark_cloud_development",
  "storm_like_structure",
  "camera_obstruction",
];

export const DatasetLabelingPage: React.FC = () => {
  const { t, tEnum } = useI18n();
  const [observations, setObservations] = useState<Observation[]>([]);
  const [selected, setSelected] = useState<Observation | null>(null);
  const [cloudLabel, setCloudLabel] = useState("cumulus");
  const [eventLabel, setEventLabel] = useState("normal");
  const [qualityLabel, setQualityLabel] = useState("usable");
  const [split, setSplit] = useState<"train" | "val" | "test">("train");
  const [brushMode, setBrushMode] = useState<"dense" | "thin" | "erase">("dense");
  const [brushSize, setBrushSize] = useState(18);
  const [statusMsg, setStatusMsg] = useState("");

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const drawingRef = useRef(false);

  useEffect(() => {
    api
      .getAnalysisHistory(60)
      .then((d) => {
        const list = (d.items || []).slice().reverse();
        setObservations(list);
        if (list.length > 0) {
          setSelected(list[0]);
          setCloudLabel(list[0].dominant_cloud_type || "cumulus");
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, c.width, c.height);
  }, [selected?.id]);

  const drawPoint = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const c = canvasRef.current;
    if (!c) return;
    const rect = c.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * c.width;
    const y = ((e.clientY - rect.top) / rect.height) * c.height;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.beginPath();
    ctx.arc(x, y, brushSize, 0, Math.PI * 2);
    if (brushMode === "erase") {
      ctx.globalCompositeOperation = "destination-out";
      ctx.fillStyle = "rgba(0,0,0,1)";
    } else {
      ctx.globalCompositeOperation = "source-over";
      ctx.fillStyle = brushMode === "dense" ? "rgba(255, 255, 255, 0.72)" : "rgba(180, 225, 255, 0.55)";
    }
    ctx.fill();
    ctx.globalCompositeOperation = "source-over";
  };

  const handleSave = async () => {
    if (!selected) return;
    const maskB64 = canvasRef.current ? canvasRef.current.toDataURL("image/png") : undefined;
    const res = await api.saveLabel({
      observation_id: selected.id,
      dataset_name: "user_camera_labels",
      cloud_type_label: cloudLabel,
      event_label: eventLabel,
      quality_label: qualityLabel,
      split,
      mask_png_base64: maskB64,
    });
    setStatusMsg(t("labeling.savedToast", { id: selected.id, ds: res.dataset_name }));
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-white">{t("labeling.title")}</h2>
          <p className="text-xs text-[#8E95A2]">{t("labeling.subtitle")}</p>
        </div>
        {statusMsg && (
          <span className="px-3 py-1 rounded-lg bg-emerald-500/15 border border-emerald-500/35 text-xs text-emerald-300">
            {statusMsg}
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-4">
        <div className="obs-panel p-3 space-y-2 max-h-[600px] overflow-y-auto">
          <div className="obs-label px-1">{t("labeling.captured", { n: observations.length })}</div>
          {observations.map((o) => (
            <button
              key={o.id}
              type="button"
              onClick={() => {
                setSelected(o);
                setCloudLabel(o.dominant_cloud_type || "cumulus");
                setStatusMsg("");
              }}
              className={`w-full text-left p-2.5 rounded-xl border text-xs transition ${
                selected?.id === o.id
                  ? "bg-white/[0.1] border-white/25 text-white"
                  : "bg-[#1A1D24]/70 border-white/[0.06] text-[#9AA1AD] hover:text-white"
              }`}
            >
              <div className="flex justify-between font-mono tabular-nums">
                <span>#{o.id}</span>
                <span>{o.cloud_coverage_pct.toFixed(0)}%</span>
              </div>
              <div className="text-[11px] text-[#646B78] truncate">
                {tEnum(o.dominant_cloud_type)} · {new Date(o.captured_at).toLocaleTimeString()}
              </div>
            </button>
          ))}
          {observations.length === 0 && (
            <p className="text-xs text-[#646B78] p-4 text-center">{t("labeling.emptyObs")}</p>
          )}
        </div>

        <div className="xl:col-span-2 obs-panel overflow-hidden">
          <div className="px-4 py-2.5 border-b border-white/[0.07] flex flex-wrap items-center justify-between gap-2 text-xs">
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => setBrushMode("dense")}
                className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-lg ${
                  brushMode === "dense" ? "obs-btn-primary" : "obs-btn-secondary"
                }`}
              >
                <Brush className="w-3.5 h-3.5" /> {t("labeling.denseBrush")}
              </button>
              <button
                type="button"
                onClick={() => setBrushMode("thin")}
                className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-lg ${
                  brushMode === "thin"
                    ? "bg-sky-500/25 border border-sky-400/50 text-sky-200 font-semibold"
                    : "obs-btn-secondary"
                }`}
              >
                <Brush className="w-3.5 h-3.5" /> {t("labeling.thinBrush")}
              </button>
              <button
                type="button"
                onClick={() => setBrushMode("erase")}
                className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-lg ${
                  brushMode === "erase"
                    ? "bg-rose-500/25 border border-rose-400/50 text-rose-200 font-semibold"
                    : "obs-btn-secondary"
                }`}
              >
                <Eraser className="w-3.5 h-3.5" /> {t("labeling.eraseBrush")}
              </button>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[#8E95A2] tabular-nums">{t("labeling.size", { n: brushSize })}</span>
              <input
                type="range"
                min={6}
                max={48}
                value={brushSize}
                onChange={(e) => setBrushSize(Number(e.target.value))}
                className="w-20 accent-[#F4F5F7]"
              />
            </div>
          </div>

          <div className="relative aspect-video bg-[#08090B] flex items-center justify-center">
            {selected ? (
              <>
                <img
                  src={`/api/analysis/image/${selected.id}?layer=raw`}
                  alt="Label target"
                  className="w-full h-full object-contain select-none pointer-events-none"
                />
                <canvas
                  ref={canvasRef}
                  width={640}
                  height={360}
                  onMouseDown={(e) => {
                    drawingRef.current = true;
                    drawPoint(e);
                  }}
                  onMouseMove={(e) => {
                    if (drawingRef.current) drawPoint(e);
                  }}
                  onMouseUp={() => {
                    drawingRef.current = false;
                  }}
                  onMouseLeave={() => {
                    drawingRef.current = false;
                  }}
                  className="absolute inset-0 w-full h-full cursor-crosshair"
                />
              </>
            ) : (
              <span className="text-xs text-[#646B78]">{t("labeling.selectHint")}</span>
            )}
          </div>
        </div>

        <div className="obs-panel p-5 space-y-3 text-xs h-fit">
          <h3 className="text-sm font-semibold text-white">{t("labeling.metaTitle")}</h3>
          <label className="block space-y-1">
            <span className="text-[#9AA1AD]">{t("labeling.cloudClass")}</span>
            <select
              value={cloudLabel}
              onChange={(e) => setCloudLabel(e.target.value)}
              className="obs-input w-full"
            >
              {CLOUD_CLASSES.map((c) => (
                <option key={c} value={c}>
                  {tEnum(c)}
                </option>
              ))}
            </select>
          </label>
          <label className="block space-y-1">
            <span className="text-[#9AA1AD]">{t("labeling.eventLabel")}</span>
            <select
              value={eventLabel}
              onChange={(e) => setEventLabel(e.target.value)}
              className="obs-input w-full"
            >
              {EVENT_LABELS.map((ev) => (
                <option key={ev} value={ev}>
                  {tEnum(ev)}
                </option>
              ))}
            </select>
          </label>
          <label className="block space-y-1">
            <span className="text-[#9AA1AD]">{t("labeling.qualityLabel")}</span>
            <select
              value={qualityLabel}
              onChange={(e) => setQualityLabel(e.target.value)}
              className="obs-input w-full"
            >
              <option value="usable">{tEnum("usable")}</option>
              <option value="blurry">{tEnum("blurry")}</option>
              <option value="obstructed">{tEnum("obstructed")}</option>
              <option value="overexposed">{tEnum("overexposed")}</option>
            </select>
          </label>
          <label className="block space-y-1">
            <span className="text-[#9AA1AD]">{t("labeling.split")}</span>
            <select
              value={split}
              onChange={(e) => setSplit(e.target.value as any)}
              className="obs-input w-full"
            >
              <option value="train">train</option>
              <option value="val">val</option>
              <option value="test">test</option>
            </select>
          </label>
          <button
            type="button"
            onClick={handleSave}
            disabled={!selected}
            className="obs-btn-primary w-full inline-flex items-center justify-center gap-1.5 px-3 py-2"
          >
            <Save className="w-3.5 h-3.5" /> {t("labeling.saveBtn")}
          </button>
        </div>
      </div>
    </div>
  );
};
