import React, { useEffect, useState } from "react";
import { BookOpen, ChevronDown, ChevronUp, Cpu, Database, Eye, Cloud, Clock, Bell, Play, RotateCcw, Sparkles, Square, Tag, Trash2 } from "lucide-react";
import { useI18n } from "../i18n/I18nContext";
import { api } from "../services/api";
import type { DatasetSummary, ModelEntry, TrainingSession } from "../types/api";

export const ModelLabPage: React.FC = () => {
  const { t, tEnum } = useI18n();
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [sessions, setSessions] = useState<TrainingSession[]>([]);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [catalog, setCatalog] = useState<any[]>([]);
  const [gpu, setGpu] = useState<any>(null);
  const [hw, setHw] = useState<any>(null);
  const [hwProfile, setHwProfile] = useState<"auto" | "low_end" | "balanced" | "high_end" | "ultra">("auto");
  const [hwDevice, setHwDevice] = useState<"auto" | "cpu" | "cuda" | "rocm" | "directml" | "mps">("auto");
  const [trainDevice, setTrainDevice] = useState<"auto" | "cpu" | "cuda" | "rocm" | "directml" | "mps">("auto");
  const [hwBusy, setHwBusy] = useState(false);
  const [showHwSpecs, setShowHwSpecs] = useState(false);
  const [hwInitialized, setHwInitialized] = useState(false);

  const [family, setFamily] = useState<"segmentation" | "classification" | "temporal" | "event">("segmentation");
  const [arch, setArch] = useState<string>("CloudSegUNet-ConvNeXt-ASPP");
  const [datasetId, setDatasetId] = useState<number>(1);
  const [epochs, setEpochs] = useState(6);
  const [batchSize, setBatchSize] = useState(4);
  const [lr, setLr] = useState(0.001);
  const [importName, setImportName] = useState("");
  const [importDir, setImportDir] = useState("");
  const [msg, setMsg] = useState("");
  const [atStatus, setAtStatus] = useState<any>(null);
  const [atDraft, setAtDraft] = useState<any>(null);
  const [atBusy, setAtBusy] = useState(false);
  const [registryFilter, setRegistryFilter] = useState<string>("all");
  const [pruneBusy, setPruneBusy] = useState(false);
  const [showGuide, setShowGuide] = useState<boolean>(true);

  const refresh = () => {
    api.getModels().then((d) => setModels(d.items || [])).catch(() => {});
    api.getTrainingStatus().then((d) => setSessions(d.sessions || [])).catch(() => {});
    api
      .getDatasets()
      .then((d) => {
        const ds = d.datasets || [];
        setDatasets(ds);
        setCatalog(d.public_catalog || []);
        if (ds.length > 0 && !ds.find((x: DatasetSummary) => x.id === datasetId)) {
          setDatasetId(ds[0].id);
        }
      })
      .catch(() => {});
    api
      .getSystemStatus()
      .then((s) => {
        setGpu(s.hardware?.gpu);
        setHw(s.hardware);
        if (!hwInitialized && s.hardware) {
          const ap = s.hardware.active_profile;
          if (ap && ["auto", "low_end", "balanced", "high_end", "ultra"].includes(ap)) {
            setHwProfile(ap);
          }
          const pd = s.hardware.gpu?.preferred_device;
          if (pd && ["auto", "cpu", "cuda", "rocm", "directml", "mps"].includes(pd)) {
            setHwDevice(pd);
            setTrainDevice(pd);
          }
          if (s.hardware.gpu?.recommended_batch_size) {
            setBatchSize(Number(s.hardware.gpu.recommended_batch_size));
          }
          setHwInitialized(true);
        }
      })
      .catch(() => {});
    api
      .getAutoTrainStatus()
      .then((st) => {
        setAtStatus(st);
        setAtDraft((prev: any) => prev ?? st.config);
      })
      .catch(() => {});
  };

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, 4000);
    return () => window.clearInterval(id);
  }, []);

  const handleToggleAutoTrain = async () => {
    if (!atStatus) return;
    setAtBusy(true);
    try {
      const next = await api.updateAutoTrainConfig({ enabled: !atStatus.enabled });
      setAtStatus(next);
      setAtDraft(next.config);
    } finally {
      setAtBusy(false);
    }
  };

  const handleSaveAutoTrainCfg = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!atDraft) return;
    setAtBusy(true);
    try {
      const next = await api.updateAutoTrainConfig(atDraft);
      setAtStatus(next);
      setAtDraft(next.config);
      setMsg(t("autotrain.savedCfg"));
    } finally {
      setAtBusy(false);
    }
  };

  const handleRunAutoTrain = async (onlyLabel: boolean) => {
    setAtBusy(true);
    try {
      await api.runAutoTrainNow({
        families: atDraft?.families,
        epochs: atDraft?.epochs,
        only_label: onlyLabel,
      });
      setMsg(t("autotrain.started"));
      refresh();
    } catch (err: any) {
      setMsg(`Error: ${err.message}`);
    } finally {
      setAtBusy(false);
    }
  };

  const handleBootstrap = async () => {
    const res = await api.bootstrapDataset();
    setDatasetId(res.id);
    setMsg(t("models.itemsCount", { name: res.name, n: res.item_count }));
    refresh();
  };

  const handleStartTrain = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await api.startTraining({
        model_family: family,
        dataset_id: datasetId,
        architecture: arch || undefined,
        epochs,
        batch_size: batchSize,
        learning_rate: lr,
        device: trainDevice,
      });
      setMsg(`${res.job_id} (${res.device})`);
      refresh();
    } catch (err: any) {
      setMsg(`Error: ${err.message}`);
    }
  };

  const handleApplyHardwareProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    setHwBusy(true);
    try {
      const res = await api.updateHardwareProfile({
        profile: hwProfile,
        device: hwDevice,
      });
      if (res.batch_size) {
        setBatchSize(Number(res.batch_size));
      }
      setTrainDevice(hwDevice);
      setHw(res.hardware);
      setGpu(res.hardware?.gpu);
      setMsg(
        t("hw.appliedToast", {
          profile: hwProfile.toUpperCase(),
          device: (res.active_device || res.device || "cpu").toUpperCase(),
          res: res.inference_resolution ?? 512,
          batch: res.batch_size ?? batchSize,
        })
      );
    } catch (err: any) {
      setMsg(`Error: ${err.message}`);
    } finally {
      setHwBusy(false);
    }
  };

  const handleImport = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await api.importDataset({ name: importName, directory_path: importDir });
      setMsg(t("models.itemsCount", { name: res.name, n: res.item_count }));
      setImportName("");
      setImportDir("");
      refresh();
    } catch (err: any) {
      setMsg(`Error: ${err.message}`);
    }
  };

  const handleDeleteModel = async (modelId: string) => {
    try {
      await api.deleteModel(modelId);
      setMsg(t("models.deletedOneToast", { id: modelId }));
      refresh();
    } catch (err: any) {
      setMsg(`Error: ${err.message}`);
    }
  };

  const handlePruneOldModels = async (keepPerFamily: number) => {
    setPruneBusy(true);
    try {
      const res = await api.pruneModels({
        keep_per_family: keepPerFamily,
        model_family: registryFilter === "all" ? null : registryFilter,
        also_prune_sessions: true,
      });
      setMsg(
        t("models.prunedToast", {
          n: res.deleted_models ?? 0,
          files: res.deleted_files ?? 0,
        })
      );
      refresh();
    } catch (err: any) {
      setMsg(`Error: ${err.message}`);
    } finally {
      setPruneBusy(false);
    }
  };

  const standbyCount = models.filter((m) => !m.is_active && !m.is_default).length;
  const filteredModels =
    registryFilter === "all"
      ? models
      : models.filter((m) => m.model_family === registryFilter);

  const modelGuideCards = [
    {
      family: "segmentation" as const,
      keyPrefix: "models.guide.seg",
      icon: Eye,
      accent: "text-sky-300 border-sky-500/30 bg-sky-500/10",
    },
    {
      family: "classification" as const,
      keyPrefix: "models.guide.cls",
      icon: Cloud,
      accent: "text-indigo-300 border-indigo-500/30 bg-indigo-500/10",
    },
    {
      family: "temporal" as const,
      keyPrefix: "models.guide.tmp",
      icon: Clock,
      accent: "text-emerald-300 border-emerald-500/30 bg-emerald-500/10",
    },
    {
      family: "event" as const,
      keyPrefix: "models.guide.evt",
      icon: Bell,
      accent: "text-amber-300 border-amber-500/30 bg-amber-500/10",
    },
  ];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-white">{t("models.title")}</h2>
          <p className="text-xs text-[#8E95A2]">
            {t("models.device")} <span className="font-mono text-[#D2D6DF]">{gpu?.device_name || "..."}</span> ·{" "}
            {t("models.safeBatch")} <span className="font-mono tabular-nums">{gpu?.recommended_batch_size ?? 4}</span>
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setShowGuide((v) => !v)}
            className="obs-btn-secondary inline-flex items-center gap-1.5 px-3 py-1.5 text-xs"
          >
            <BookOpen className="w-3.5 h-3.5 text-emerald-400" />
            {showGuide ? t("models.guideHide") : t("models.guideShow")}
            {showGuide ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>
          <button
            type="button"
            onClick={handleBootstrap}
            className="obs-btn-secondary inline-flex items-center gap-1.5 px-3 py-1.5 text-xs"
          >
            <Database className="w-3.5 h-3.5 text-sky-400" /> {t("models.prepareBaseline")}
          </button>
        </div>
      </div>

      {msg && <div className="obs-inset px-4 py-2 text-xs text-sky-200">{msg}</div>}

      {/* Universal Cross-Platform Hardware Compatibility, CPU/GPU Engine & Performance Preset Selector */}
      <form onSubmit={handleApplyHardwareProfile} className="obs-panel p-5 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Cpu className="w-4 h-4 text-sky-400" /> {t("hw.title")}
            </h3>
            <p className="text-xs text-[#8E95A2] mt-0.5">{t("hw.subtitle")}</p>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono">
            {hw && (
              <>
                <span className="px-2.5 py-1 rounded-lg bg-[#0B0C0E] border border-white/[0.08] text-[#D2D6DF]">
                  {t("hw.detectedHost")}{" "}
                  <strong className="text-white">
                    {hw.os_name || "OS"} ({hw.architecture || "x64"}) · {hw.cpu_cores ?? "?"}T · {hw.ram_total_gb ?? "?"} GB RAM
                  </strong>
                </span>
                <span className="px-2.5 py-1 rounded-lg bg-sky-500/15 border border-sky-400/35 text-sky-200">
                  {t("hw.detectedTier")} <strong>{(hw.detected_tier || "balanced").toUpperCase()}</strong>
                </span>
                <span
                  className={`px-2.5 py-1 rounded-lg border ${
                    hw.meets_recommended_requirements
                      ? "bg-emerald-500/15 border-emerald-500/35 text-emerald-300"
                      : hw.meets_minimum_requirements !== false
                      ? "bg-amber-500/15 border-amber-500/35 text-amber-200"
                      : "bg-rose-500/15 border-rose-500/35 text-rose-300"
                  }`}
                >
                  {hw.meets_recommended_requirements
                    ? t("hw.meetsRec")
                    : hw.meets_minimum_requirements !== false
                    ? t("hw.meetsMin")
                    : t("hw.belowMin")}
                </span>
              </>
            )}
            <button
              type="button"
              onClick={() => setShowHwSpecs((v) => !v)}
              className="obs-btn-secondary inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-sans"
            >
              <span>{t("hw.specsToggle")}</span>
              {showHwSpecs ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-12 gap-3 items-end text-xs">
          <label className="md:col-span-5 space-y-1">
            <span className="text-[#9AA1AD]">{t("hw.selectDevice")}</span>
            <select
              value={hwDevice}
              onChange={(e) => setHwDevice(e.target.value as any)}
              className="obs-input w-full"
            >
              <option value="auto">{t("hw.dev.auto")}</option>
              <option value="cpu">{t("hw.dev.cpu")}</option>
              <option value="cuda">
                {t("hw.dev.cuda")} {gpu?.cuda_available && !gpu?.rocm_available ? `(${gpu.device_name})` : ""}
              </option>
              <option value="rocm">
                {t("hw.dev.rocm")} {gpu?.rocm_available ? `(${gpu.device_name})` : ""}
              </option>
              <option value="directml">
                {t("hw.dev.directml")} {gpu?.directml_available ? "(Ready)" : ""}
              </option>
              {(gpu?.mps_available || hw?.os_name === "Darwin") && (
                <option value="mps">{t("hw.dev.mps")}</option>
              )}
            </select>
          </label>

          <label className="md:col-span-5 space-y-1">
            <span className="text-[#9AA1AD]">{t("hw.selectProfile")}</span>
            <select
              value={hwProfile}
              onChange={(e) => {
                const nextP = e.target.value as "auto" | "low_end" | "balanced" | "high_end" | "ultra";
                setHwProfile(nextP);
                if (nextP === "low_end") setHwDevice("cpu");
                else if (nextP === "high_end" || nextP === "ultra") {
                  setHwDevice(gpu?.cuda_available ? "cuda" : "auto");
                }
              }}
              className="obs-input w-full"
            >
              <option value="auto">
                {t("hw.prof.auto", { tier: (hw?.detected_tier || "balanced").toUpperCase() })}
              </option>
              <option value="low_end">{t("hw.prof.low_end")}</option>
              <option value="balanced">{t("hw.prof.balanced")}</option>
              <option value="high_end">{t("hw.prof.high_end")}</option>
              <option value="ultra">{t("hw.prof.ultra")}</option>
            </select>
          </label>

          <div className="md:col-span-2">
            <button
              type="submit"
              disabled={hwBusy}
              className="obs-btn-primary w-full px-3 py-2 font-medium"
            >
              {t("hw.applyBtn")}
            </button>
          </div>
        </div>

        {showHwSpecs && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2 border-t border-white/[0.06] text-[11px] leading-relaxed">
            <div className="obs-inset p-3 space-y-1 border border-white/[0.05]">
              <div className="font-semibold text-amber-300">{t("hw.minTitle")}</div>
              <div className="text-[#9AA1AD]">{t("hw.minDesc")}</div>
            </div>
            <div className="obs-inset p-3 space-y-1 border border-sky-500/25">
              <div className="font-semibold text-sky-300">{t("hw.recTitle")}</div>
              <div className="text-[#9AA1AD]">{t("hw.recDesc")}</div>
            </div>
            <div className="obs-inset p-3 space-y-1 border border-emerald-500/25">
              <div className="font-semibold text-emerald-300">{t("hw.highTitle")}</div>
              <div className="text-[#9AA1AD]">{t("hw.highDesc")}</div>
            </div>
          </div>
        )}
      </form>

      {/* 4 AI Models Overview & Use Case Guide */}
      {showGuide && (
        <div className="obs-panel p-5 space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-sky-400" /> {t("models.guideTitle")}
            </h3>
            <p className="text-xs text-[#8E95A2] mt-0.5">{t("models.guideSubtitle")}</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {modelGuideCards.map((card) => {
              const Icon = card.icon;
              const activeModel = models.find((m) => m.model_family === card.family && m.is_active);
              return (
                <div
                  key={card.family}
                  className="obs-inset p-3.5 space-y-2.5 flex flex-col justify-between border border-white/[0.06]"
                >
                  <div className="space-y-2">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className={`p-1.5 rounded-lg border ${card.accent}`}>
                          <Icon className="w-3.5 h-3.5" />
                        </span>
                        <span className="text-xs font-semibold text-white">
                          {t(`${card.keyPrefix}.title`)}
                        </span>
                      </div>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-mono border ${card.accent}`}>
                        {t(`${card.keyPrefix}.badge`)}
                      </span>
                    </div>

                    <div className="text-[11px] leading-relaxed space-y-1.5">
                      <div>
                        <span className="font-semibold text-[#D2D6DF]">{t("models.guideWhatItDoes")} </span>
                        <span className="text-[#9AA1AD]">{t(`${card.keyPrefix}.what`)}</span>
                      </div>
                      <div>
                        <span className="font-semibold text-sky-300">{t("models.guideUseCase")} </span>
                        <span className="text-[#C5CBD5]">{t(`${card.keyPrefix}.use`)}</span>
                      </div>
                      <div className="pt-1 border-t border-white/[0.06]">
                        <span className="font-semibold text-emerald-300">{t("models.guideMetrics")} </span>
                        <span className="font-mono text-[10.5px] text-[#8E95A2]">
                          {t(`${card.keyPrefix}.metrics`)}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="pt-2 border-t border-white/[0.06] flex flex-wrap items-center justify-between gap-2 text-[11px]">
                    <div className="font-mono text-[#8E95A2] truncate">
                      {t("models.guideActiveNow")}{" "}
                      <span className="text-emerald-300 font-semibold">
                        {activeModel
                          ? activeModel.model_id
                          : card.family === "segmentation"
                          ? `radiometric-physics-v1 (${t("models.builtInTag")})`
                          : t("models.builtInTag")}
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        setFamily(card.family);
                        setRegistryFilter(card.family);
                      }}
                      className="obs-btn-secondary px-2 py-0.5 text-[10px] font-mono"
                    >
                      {t("models.guideSelectToFilter")} ({tEnum(card.family)})
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Location Auto-Train & Auto-Label Card */}
      <form onSubmit={handleSaveAutoTrainCfg} className="obs-panel p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-emerald-400" /> {t("autotrain.title")}
            </h3>
            <p className="text-xs text-[#8E95A2]">{t("autotrain.subtitle")}</p>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <button
              type="button"
              disabled={atBusy || !atStatus}
              onClick={handleToggleAutoTrain}
              className={`px-3 py-1.5 rounded-lg border font-medium transition ${
                atStatus?.enabled
                  ? "bg-emerald-500/15 border-emerald-500/35 text-emerald-300"
                  : "obs-btn-secondary"
              }`}
            >
              {atStatus?.enabled ? t("autotrain.modeOn") : t("autotrain.modeOff")}
            </button>
            <button
              type="button"
              disabled={atBusy || atStatus?.running || (atStatus?.eligible_local_observations ?? 0) === 0}
              onClick={() => handleRunAutoTrain(true)}
              className="obs-btn-secondary inline-flex items-center gap-1.5 px-3 py-1.5"
            >
              <Tag className="w-3.5 h-3.5" /> {t("autotrain.labelOnlyBtn")}
            </button>
            {atStatus?.running ? (
              <button
                type="button"
                onClick={() => api.stopAutoTrain().then(refresh)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/35 border border-rose-500/35 text-rose-200 font-medium"
              >
                <Square className="w-3.5 h-3.5" /> {t("autotrain.stopBtn")}
              </button>
            ) : (
              <button
                type="button"
                disabled={atBusy || (atStatus?.eligible_local_observations ?? 0) === 0}
                onClick={() => handleRunAutoTrain(false)}
                className="obs-btn-primary inline-flex items-center gap-1.5 px-3.5 py-1.5"
              >
                <Sparkles className="w-3.5 h-3.5" /> {t("autotrain.runBtn")}
              </button>
            )}
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
          <div className="obs-inset p-2.5">
            <div className="obs-label">{t("autotrain.eligible", { n: "" }).trim()}</div>
            <div className="font-mono tabular-nums text-sm font-semibold text-white mt-0.5">
              {atStatus?.eligible_local_observations ?? 0}
            </div>
          </div>
          <div className="obs-inset p-2.5">
            <div className="obs-label">Local Labeled</div>
            <div className="font-mono tabular-nums text-sm font-semibold text-sky-300 mt-0.5">
              {atStatus?.last_summary?.local_items ?? 0}
            </div>
          </div>
          <div className="obs-inset p-2.5">
            <div className="obs-label">Confirmed Events</div>
            <div className="font-mono tabular-nums text-sm font-semibold text-emerald-300 mt-0.5">
              {atStatus?.last_summary?.confirmed_event_items ?? 0}
            </div>
          </div>
          <div className="obs-inset p-2.5">
            <div className="obs-label">Replay Anchor</div>
            <div className="font-mono tabular-nums text-sm font-semibold text-[#C5CBD5] mt-0.5">
              {atStatus?.last_summary?.replay_items ?? 0}
            </div>
          </div>
        </div>

        {atDraft && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 items-end text-xs">
            <label className="lg:col-span-2 space-y-1">
              <span className="text-[#9AA1AD]">{t("autotrain.intervalMin")}</span>
              <input
                type="number"
                min={5}
                max={1440}
                value={atDraft.interval_minutes ?? 60}
                onChange={(e) => setAtDraft({ ...atDraft, interval_minutes: Number(e.target.value) })}
                className="obs-input w-full font-mono tabular-nums"
              />
            </label>
            <label className="lg:col-span-2 space-y-1">
              <span className="text-[#9AA1AD]">{t("autotrain.minNew")}</span>
              <input
                type="number"
                min={2}
                max={500}
                value={atDraft.min_new_observations ?? 12}
                onChange={(e) => setAtDraft({ ...atDraft, min_new_observations: Number(e.target.value) })}
                className="obs-input w-full font-mono tabular-nums"
              />
            </label>
            <label className="lg:col-span-2 space-y-1">
              <span className="text-[#9AA1AD]">{t("autotrain.maxSamples")}</span>
              <input
                type="number"
                min={4}
                max={1000}
                value={atDraft.max_local_samples ?? 80}
                onChange={(e) => setAtDraft({ ...atDraft, max_local_samples: Number(e.target.value) })}
                className="obs-input w-full font-mono tabular-nums"
              />
            </label>
            <label className="lg:col-span-1 space-y-1">
              <span className="text-[#9AA1AD]">{t("autotrain.epochs")}</span>
              <input
                type="number"
                min={1}
                max={30}
                value={atDraft.epochs ?? 3}
                onChange={(e) => setAtDraft({ ...atDraft, epochs: Number(e.target.value) })}
                className="obs-input w-full font-mono tabular-nums"
              />
            </label>
            <div className="lg:col-span-3 space-y-1.5">
              <div className="flex flex-wrap gap-2">
                {(["segmentation", "classification", "event", "temporal"] as const).map((fam) => {
                  const checked = (atDraft.families || []).includes(fam);
                  return (
                    <label key={fam} className="inline-flex items-center gap-1 text-[#D2D6DF] cursor-pointer">
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={(e) => {
                          const cur: string[] = atDraft.families || [];
                          setAtDraft({
                            ...atDraft,
                            families: e.target.checked ? [...cur, fam] : cur.filter((x) => x !== fam),
                          });
                        }}
                      />
                      <span>{tEnum(fam)}</span>
                    </label>
                  );
                })}
              </div>
              <label className="flex items-center gap-1.5 text-[#8E95A2] cursor-pointer">
                <input
                  type="checkbox"
                  checked={!!atDraft.include_baseline_replay}
                  onChange={(e) => setAtDraft({ ...atDraft, include_baseline_replay: e.target.checked })}
                />
                <span>{t("autotrain.replayOpt")}</span>
              </label>
              <label className="flex items-center gap-1.5 text-[#8E95A2] cursor-pointer">
                <input
                  type="checkbox"
                  checked={!!atDraft.warm_start}
                  onChange={(e) => setAtDraft({ ...atDraft, warm_start: e.target.checked })}
                />
                <span>{t("autotrain.warmStartOpt")}</span>
              </label>
            </div>
            <div className="lg:col-span-2">
              <button type="submit" disabled={atBusy} className="obs-btn-secondary w-full px-3 py-2">
                {t("autotrain.saveCfg")}
              </button>
            </div>
          </div>
        )}
      </form>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <form onSubmit={handleStartTrain} className="obs-panel p-5 space-y-3">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <Cpu className="w-4 h-4 text-sky-400" /> {t("models.startTitle")}
          </h3>
          <div className="space-y-2.5 text-xs">
            <label className="block space-y-1">
              <span className="text-[#9AA1AD]">{t("models.family")}</span>
              <select
                value={family}
                onChange={(e) => {
                  const next = e.target.value as "segmentation" | "classification" | "temporal" | "event";
                  setFamily(next);
                  setArch(
                    next === "segmentation"
                      ? "CloudSegUNet-ConvNeXt-ASPP"
                      : next === "classification"
                      ? "CloudTypeClassifier-EfficientNetB0"
                      : ""
                  );
                }}
                className="obs-input w-full"
              >
                <option value="segmentation">{t("models.fam.seg")}</option>
                <option value="classification">{t("models.fam.cls")}</option>
                <option value="temporal">{t("models.fam.tmp")}</option>
                <option value="event">{t("models.fam.evt")}</option>
              </select>
            </label>
            <div className="obs-inset p-2.5 text-[11px] leading-relaxed text-[#9AA1AD]">
              {t(
                family === "segmentation"
                  ? "models.guide.seg.use"
                  : family === "classification"
                  ? "models.guide.cls.use"
                  : family === "temporal"
                  ? "models.guide.tmp.use"
                  : "models.guide.evt.use"
              )}
            </div>
            <label className="block space-y-1">
              <span className="text-[#9AA1AD]">{t("models.dataset")}</span>
              <select
                value={datasetId}
                onChange={(e) => setDatasetId(Number(e.target.value))}
                className="obs-input w-full"
              >
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {t("models.itemsCount", { name: d.name, n: d.item_count })}
                  </option>
                ))}
                {datasets.length === 0 && <option value={0}>{t("models.noDatasetOpt")}</option>}
              </select>
            </label>
            {(family === "segmentation" || family === "classification") && (
              <label className="block space-y-1">
                <span className="text-[#9AA1AD]">Architecture</span>
                <select
                  value={arch}
                  onChange={(e) => setArch(e.target.value)}
                  className="obs-input w-full"
                >
                  {family === "segmentation" ? (
                    <>
                      <option value="CloudSegUNet-ConvNeXt-ASPP">ConvNeXt-Tiny + ASPP + scSE Attention (Pro)</option>
                      <option value="CloudSegUNet-EfficientNetB0">EfficientNet-B0 UNet (Balanced)</option>
                      <option value="CloudSegUNet-MobileNetV3">MobileNetV3-Small UNet (Fast)</option>
                      <option value="CloudSegUNet-Mobile">Lightweight Depthwise UNet (Minimal)</option>
                    </>
                  ) : (
                    <>
                      <option value="CloudTypeClassifier-EfficientNetB0">EfficientNet-B0 + Gated Fusion (Pretrained)</option>
                      <option value="CloudTypeClassifier-ConvNeXt">ConvNeXt-Tiny + Gated Fusion (Pro)</option>
                      <option value="CloudTypeClassifier-Fusion">Compact CNN + Fusion (Legacy)</option>
                    </>
                  )}
                </select>
              </label>
            )}
            <label className="block space-y-1">
              <span className="text-[#9AA1AD]">{t("hw.trainDeviceLabel")}</span>
              <select
                value={trainDevice}
                onChange={(e) => setTrainDevice(e.target.value as any)}
                className="obs-input w-full"
              >
                <option value="auto">{t("hw.dev.auto")}</option>
                <option value="cpu">{t("hw.dev.cpu")}</option>
                <option value="cuda">{t("hw.dev.cuda")}</option>
                <option value="rocm">{t("hw.dev.rocm")}</option>
                <option value="directml">{t("hw.dev.directml")}</option>
                {(gpu?.mps_available || hw?.os_name === "Darwin") && (
                  <option value="mps">{t("hw.dev.mps")}</option>
                )}
              </select>
            </label>
            <div className="grid grid-cols-3 gap-2">
              <label className="block space-y-1">
                <span className="text-[#9AA1AD]">{t("models.epochs")}</span>
                <input
                  type="number"
                  min={1}
                  max={200}
                  value={epochs}
                  onChange={(e) => setEpochs(Number(e.target.value))}
                  className="obs-input w-full font-mono tabular-nums"
                />
              </label>
              <label className="block space-y-1">
                <span className="text-[#9AA1AD]">{t("models.batch")}</span>
                <input
                  type="number"
                  min={1}
                  max={32}
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  className="obs-input w-full font-mono tabular-nums"
                />
              </label>
              <label className="block space-y-1">
                <span className="text-[#9AA1AD]">{t("models.lr")}</span>
                <input
                  type="number"
                  step="0.0002"
                  value={lr}
                  onChange={(e) => setLr(Number(e.target.value))}
                  className="obs-input w-full font-mono tabular-nums"
                />
              </label>
            </div>
            <button
              type="submit"
              disabled={datasets.length === 0}
              className="obs-btn-primary w-full inline-flex items-center justify-center gap-1.5 px-3 py-2"
            >
              <Play className="w-3.5 h-3.5" /> {t("models.launchTrain")}
            </button>
          </div>
        </form>

        <div className="xl:col-span-2 obs-panel p-5 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-sm font-semibold text-white">{t("models.sessionsTitle")}</h3>
            <div className="flex items-center gap-1.5">
              {sessions.length > 0 && (
                <button
                  type="button"
                  onClick={() => api.clearTrainingSessions().then(refresh)}
                  className="obs-btn-secondary inline-flex items-center gap-1 px-2.5 py-1 text-xs"
                >
                  <Trash2 className="w-3 h-3 text-[#8E95A2]" /> {t("models.clearSessionsBtn")}
                </button>
              )}
              <button
                type="button"
                onClick={async () => {
                  try {
                    const res = await api.stopTraining();
                    const n = res.stopped_count ?? (res.stop_requested_for?.length || 0);
                    setMsg(n > 0 ? `Stopped ${n} active training session(s).` : "No active training sessions running.");
                    refresh();
                  } catch (err: any) {
                    setMsg(`Error: ${err.message}`);
                  }
                }}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-rose-500/15 hover:bg-rose-500/25 text-rose-300 border border-rose-500/30 text-xs"
              >
                <Square className="w-3 h-3" /> {t("models.stopRunning")}
              </button>
            </div>
          </div>
          <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
            {sessions.map((s) => (
              <div key={s.job_id} className="obs-inset p-3 text-xs space-y-1.5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-mono font-semibold text-white">
                    {s.job_id} <span className="text-[#8E95A2]">({s.architecture})</span>
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded font-mono text-[10px] uppercase ${
                      s.status === "completed"
                        ? "bg-emerald-500/20 text-emerald-300"
                        : s.status === "running"
                        ? "bg-sky-500/20 text-sky-300"
                        : s.status === "failed"
                        ? "bg-rose-500/20 text-rose-300"
                        : "bg-white/10 text-[#9AA1AD]"
                    }`}
                  >
                    {tEnum(s.status)} · Ep {s.current_epoch}/{s.total_epochs}
                  </span>
                </div>
                <div className="h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#F4F5F7] transition-all"
                    style={{ width: `${Math.min(100, (s.current_epoch / Math.max(1, s.total_epochs)) * 100)}%` }}
                  />
                </div>
                <div className="flex justify-between text-[11px] font-mono tabular-nums text-[#8E95A2]">
                  <span>{t("models.device")} {s.device_used}</span>
                  <span>{t("models.bestMetric")} {s.best_val_metric}</span>
                </div>
                {s.error_message && <div className="text-rose-400">{s.error_message}</div>}
              </div>
            ))}
            {sessions.length === 0 && (
              <p className="text-xs text-[#646B78] py-6 text-center">{t("models.noSessions")}</p>
            )}
          </div>
        </div>
      </div>

      <div className="obs-panel p-5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-semibold text-white">{t("models.registryTitle")}</h3>
            {standbyCount > 0 && (
              <p className="text-[11px] text-[#8E95A2] mt-0.5">
                {t("models.standbyCount", { n: standbyCount })}
              </p>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            {standbyCount > 0 && (
              <>
                <button
                  type="button"
                  disabled={pruneBusy}
                  onClick={() => handlePruneOldModels(2)}
                  className="obs-btn-secondary inline-flex items-center gap-1 px-2.5 py-1 text-[11px]"
                >
                  <Trash2 className="w-3 h-3 text-amber-400" /> {t("models.pruneKeepBackup")}
                </button>
                <button
                  type="button"
                  disabled={pruneBusy}
                  onClick={() => handlePruneOldModels(1)}
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 border border-rose-500/35 text-rose-200 text-[11px] font-medium transition disabled:opacity-40"
                >
                  <Trash2 className="w-3 h-3" /> {t("models.pruneKeepActive")}
                </button>
              </>
            )}
            {(["segmentation", "classification", "temporal", "event"] as const).map((fam) => (
              <button
                key={fam}
                type="button"
                onClick={() => api.rollbackModel(fam).then(refresh).catch((e) => setMsg(e.message))}
                className="obs-btn-secondary inline-flex items-center gap-1 px-2.5 py-1 text-[11px]"
              >
                <RotateCcw className="w-3 h-3" /> {t("models.rollback")} {tEnum(fam)}
              </button>
            ))}
          </div>
        </div>

        {/* Family Filter Bar */}
        <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
          <button
            type="button"
            onClick={() => setRegistryFilter("all")}
            className={`px-2.5 py-1 rounded-lg border font-mono transition ${
              registryFilter === "all"
                ? "bg-white/[0.12] border-white/25 text-white font-semibold"
                : "bg-[#0B0C0E] border-white/[0.07] text-[#8E95A2] hover:text-white"
            }`}
          >
            {t("models.filterAll", { n: models.length + 1 })}
          </button>
          {(["segmentation", "classification", "temporal", "event"] as const).map((fam) => {
            const count = models.filter((m) => m.model_family === fam).length;
            return (
              <button
                key={fam}
                type="button"
                onClick={() => setRegistryFilter(fam)}
                className={`px-2.5 py-1 rounded-lg border font-mono transition ${
                  registryFilter === fam
                    ? "bg-sky-500/20 border-sky-400/45 text-sky-200 font-semibold"
                    : "bg-[#0B0C0E] border-white/[0.07] text-[#8E95A2] hover:text-white"
                }`}
              >
                {tEnum(fam)} ({fam === "segmentation" ? t("models.builtInFilterHint", { n: count }) : count})
              </button>
            );
          })}
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="text-[#8E95A2] border-b border-white/[0.08]">
              <tr>
                <th className="py-2 pr-3">{t("models.col.id")}</th>
                <th className="py-2 pr-3">{t("models.col.family")}</th>
                <th className="py-2 pr-3">{t("models.col.arch")}</th>
                <th className="py-2 pr-3">{t("models.col.val")}</th>
                <th className="py-2 pr-3">{t("models.col.status")}</th>
                <th className="py-2 pr-3">{t("models.col.action")}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.05]">
              {(registryFilter === "all" || registryFilter === "segmentation") && (() => {
                const segNeuralActive = models.some((m) => m.model_family === "segmentation" && m.is_active);
                return (
                  <tr className="bg-sky-500/[0.03]">
                    <td className="py-2 pr-3 font-mono text-white">
                      radiometric-physics-v1{" "}
                      <span className="ml-1 px-1.5 py-0.5 rounded bg-sky-500/15 border border-sky-400/30 text-sky-300 text-[9.5px]">
                        {t("models.builtInTag")}
                      </span>
                    </td>
                    <td className="py-2 pr-3">{tEnum("segmentation")}</td>
                    <td className="py-2 pr-3 text-[#C5CBD5]">{t("models.builtInArch")}</td>
                    <td className="py-2 pr-3 font-mono text-[11px] text-[#9AA1AD]">
                      {t("models.builtInMetrics")}
                    </td>
                    <td className="py-2 pr-3">
                      {!segNeuralActive ? (
                        <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 text-[10px] font-mono">
                          {t("models.builtInActive")}
                        </span>
                      ) : (
                        <span className="text-[#646B78]">{t("models.standby")}</span>
                      )}
                    </td>
                    <td className="py-2 pr-3 text-[11px] font-mono text-[#8E95A2]">
                      {segNeuralActive ? (
                        <button
                          type="button"
                          onClick={() => api.activateModel("radiometric-physics-v1").then(refresh)}
                          className="obs-btn-primary px-2.5 py-1 text-[11px] font-sans"
                        >
                          {t("models.activate")}
                        </button>
                      ) : (
                        t("models.builtInActionHint")
                      )}
                    </td>
                  </tr>
                );
              })()}
              {filteredModels.map((m) => (
                <tr key={m.model_id}>
                  <td className="py-2 pr-3 font-mono text-white">
                    {m.model_id}
                    {m.is_default && (
                      <span className="ml-1.5 px-1.5 py-0.5 rounded bg-amber-500/15 border border-amber-400/30 text-amber-300 text-[9.5px]">
                        DEFAULT
                      </span>
                    )}
                  </td>
                  <td className="py-2 pr-3">{tEnum(m.model_family)}</td>
                  <td className="py-2 pr-3 text-[#C5CBD5]">{m.architecture}</td>
                  <td className="py-2 pr-3 font-mono tabular-nums text-[11px] text-[#9AA1AD]">
                    {Object.entries(m.metrics)
                      .map(([k, v]) => `${k}=${v}`)
                      .join(" · ") || "—"}
                  </td>
                  <td className="py-2 pr-3">
                    {m.is_active ? (
                      <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 text-[10px] font-mono">
                        {t("models.active")}
                      </span>
                    ) : (
                      <span className="text-[#646B78]">{t("models.standby")}</span>
                    )}
                  </td>
                  <td className="py-2 pr-3">
                    {m.is_active ? (
                      m.model_family === "segmentation" && (
                        <button
                          type="button"
                          onClick={() => api.deactivateModelFamily("segmentation").then(refresh)}
                          className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-amber-500/15 hover:bg-amber-500/25 border border-amber-400/35 text-amber-200 text-[11px] font-medium transition"
                        >
                          {t("models.deactivate")}
                        </button>
                      )
                    ) : (
                      <div className="inline-flex items-center gap-1.5">
                        <button
                          type="button"
                          onClick={() => api.activateModel(m.model_id).then(refresh)}
                          className="obs-btn-primary px-2.5 py-1 text-[11px]"
                        >
                          {t("models.activate")}
                        </button>
                        {!m.is_default && (
                          <button
                            type="button"
                            onClick={() => handleDeleteModel(m.model_id)}
                            title={t("models.deleteOne")}
                            className="inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-rose-500/15 hover:bg-rose-500/30 border border-rose-500/30 text-rose-300 text-[11px] transition"
                          >
                            <Trash2 className="w-3 h-3" /> {t("models.deleteOne")}
                          </button>
                        )}
                      </div>
                    )}
                  </td>
                </tr>
              ))}
              {filteredModels.length === 0 && registryFilter !== "all" && registryFilter !== "segmentation" && (
                <tr>
                  <td colSpan={6} className="py-6 text-center text-[#646B78]">
                    {t("models.noModels")}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <form onSubmit={handleImport} className="obs-panel p-5 space-y-3 text-xs">
          <h3 className="text-sm font-semibold text-white">{t("models.importTitle")}</h3>
          <p className="text-[#8E95A2]">{t("models.importDesc")}</p>
          <input
            type="text"
            placeholder={t("models.importNamePh")}
            value={importName}
            onChange={(e) => setImportName(e.target.value)}
            required
            className="obs-input w-full"
          />
          <input
            type="text"
            placeholder={t("models.importDirPh")}
            value={importDir}
            onChange={(e) => setImportDir(e.target.value)}
            required
            className="obs-input w-full font-mono"
          />
          <button type="submit" className="obs-btn-secondary px-3 py-2">
            {t("models.importBtn")}
          </button>
        </form>

        <div className="obs-panel p-5 space-y-2.5">
          <h3 className="text-sm font-semibold text-white">{t("models.catalogTitle")}</h3>
          <div className="space-y-2 max-h-56 overflow-y-auto pr-1 text-xs">
            {catalog.map((c) => (
              <div key={c.id} className="obs-inset p-2.5">
                <div className="font-semibold text-white">{c.name}</div>
                <div className="text-[11px] text-[#8E95A2]">
                  {c.citation} · {c.samples} · {c.license}
                </div>
                <div className="text-[#C5CBD5] mt-1">{c.notes}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
