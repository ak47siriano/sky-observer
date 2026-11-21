import React, { useEffect, useRef, useState } from "react";
import { Brush, Eraser, Save, Square, Trash2 } from "lucide-react";
import { useI18n } from "../i18n/I18nContext";
import { api } from "../services/api";

interface Props {
  imageUrl: string;
  onSaved?: () => void;
}

export const ExclusionMaskPainter: React.FC<Props> = ({ imageUrl, onSaved }) => {
  const { t } = useI18n();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const drawingRef = useRef(false);
  const boxStartRef = useRef<{ x: number; y: number } | null>(null);
  const snapshotBeforeBoxRef = useRef<ImageData | null>(null);

  const [tool, setTool] = useState<"brush" | "box" | "erase">("brush");
  const [brushSize, setBrushSize] = useState(24);
  const [statusMsg, setStatusMsg] = useState("");
  const [saving, setSaving] = useState(false);

  // Load previously saved painted exclusion mask onto canvas
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const img = new Image();
    img.onload = () => {
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.drawImage(img, 0, 0, c.width, c.height);
    };
    img.src = `/api/camera/exclusion-mask.png?t=${Date.now()}`;
  }, []);

  const getCoords = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const c = canvasRef.current!;
    const rect = c.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * c.width,
      y: ((e.clientY - rect.top) / rect.height) * c.height,
    };
  };

  const paintCircle = (x: number, y: number, eraseMode: boolean) => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    ctx.beginPath();
    ctx.arc(x, y, brushSize, 0, Math.PI * 2);
    if (eraseMode) {
      ctx.globalCompositeOperation = "destination-out";
      ctx.fillStyle = "rgba(0,0,0,1)";
    } else {
      ctx.globalCompositeOperation = "source-over";
      ctx.fillStyle = "rgba(244, 63, 94, 0.65)";
    }
    ctx.fill();
    ctx.globalCompositeOperation = "source-over";
  };

  const onMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const c = canvasRef.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;
    const { x, y } = getCoords(e);
    drawingRef.current = true;
    if (tool === "box") {
      boxStartRef.current = { x, y };
      snapshotBeforeBoxRef.current = ctx.getImageData(0, 0, c.width, c.height);
    } else {
      paintCircle(x, y, tool === "erase");
    }
  };

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current) return;
    const c = canvasRef.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;
    const { x, y } = getCoords(e);
    if (tool === "box" && boxStartRef.current && snapshotBeforeBoxRef.current) {
      ctx.putImageData(snapshotBeforeBoxRef.current, 0, 0);
      const x0 = Math.min(boxStartRef.current.x, x);
      const y0 = Math.min(boxStartRef.current.y, y);
      const rw = Math.abs(x - boxStartRef.current.x);
      const rh = Math.abs(y - boxStartRef.current.y);
      ctx.fillStyle = "rgba(244, 63, 94, 0.65)";
      ctx.fillRect(x0, y0, rw, rh);
    } else {
      paintCircle(x, y, tool === "erase");
    }
  };

  const finishStroke = () => {
    drawingRef.current = false;
    boxStartRef.current = null;
    snapshotBeforeBoxRef.current = null;
  };

  const handleSave = async () => {
    if (!canvasRef.current) return;
    setSaving(true);
    try {
      const b64 = canvasRef.current.toDataURL("image/png");
      const res = await api.savePaintedExclusionMask(b64);
      setStatusMsg(t("exclude.savedPct", { pct: res.excluded_pct }));
      onSaved?.();
      setTimeout(() => setStatusMsg(""), 3500);
    } finally {
      setSaving(false);
    }
  };

  const handleClear = async () => {
    const c = canvasRef.current;
    const ctx = c?.getContext("2d");
    if (c && ctx) ctx.clearRect(0, 0, c.width, c.height);
    await api.clearPaintedExclusionMask();
    setStatusMsg(t("exclude.cleared"));
    onSaved?.();
    setTimeout(() => setStatusMsg(""), 3000);
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            onClick={() => setTool("brush")}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg font-medium transition ${
              tool === "brush"
                ? "bg-rose-500/25 text-rose-200 border border-rose-500/50"
                : "obs-btn-secondary"
            }`}
          >
            <Brush className="w-3.5 h-3.5" /> {t("exclude.brush")}
          </button>
          <button
            type="button"
            onClick={() => setTool("box")}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg font-medium transition ${
              tool === "box"
                ? "bg-rose-500/25 text-rose-200 border border-rose-500/50"
                : "obs-btn-secondary"
            }`}
          >
            <Square className="w-3.5 h-3.5" /> {t("exclude.box")}
          </button>
          <button
            type="button"
            onClick={() => setTool("erase")}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg font-medium transition ${
              tool === "erase"
                ? "bg-emerald-500/25 text-emerald-200 border border-emerald-500/50"
                : "obs-btn-secondary"
            }`}
          >
            <Eraser className="w-3.5 h-3.5" /> {t("exclude.erase")}
          </button>
          <div className="flex items-center gap-2 pl-2">
            <span className="text-[#8E95A2] tabular-nums">{t("labeling.size", { n: brushSize })}</span>
            <input
              type="range"
              min={6}
              max={64}
              value={brushSize}
              onChange={(e) => setBrushSize(Number(e.target.value))}
              className="w-24 accent-rose-500"
            />
          </div>
        </div>

        <div className="flex items-center gap-2">
          {statusMsg && <span className="text-emerald-400 font-mono text-[11px]">{statusMsg}</span>}
          <button
            type="button"
            onClick={handleClear}
            className="obs-btn-secondary inline-flex items-center gap-1 px-2.5 py-1.5"
          >
            <Trash2 className="w-3.5 h-3.5" /> {t("exclude.clear")}
          </button>
          <button
            type="button"
            disabled={saving}
            onClick={handleSave}
            className="obs-btn-primary inline-flex items-center gap-1.5 px-3 py-1.5"
          >
            <Save className="w-3.5 h-3.5" /> {t("exclude.save")}
          </button>
        </div>
      </div>

      <p className="text-xs text-[#8E95A2]">{t("exclude.hint")}</p>

      <div className="relative aspect-video bg-[#08090B] border border-white/[0.08] rounded-xl overflow-hidden">
        <img
          src={imageUrl}
          onError={(e) => {
            (e.currentTarget as HTMLImageElement).style.display = "none";
          }}
          alt="Exclusion paint canvas"
          className="w-full h-full object-contain select-none pointer-events-none"
        />
        <canvas
          ref={canvasRef}
          width={640}
          height={360}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={finishStroke}
          onMouseLeave={finishStroke}
          className="absolute inset-0 w-full h-full cursor-crosshair"
        />
      </div>
    </div>
  );
};
