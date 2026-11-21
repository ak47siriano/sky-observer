import React, { useEffect, useState } from "react";
import { Bell, Check, Image, RotateCcw, Save, Sparkles, Trash2, X } from "lucide-react";
import { useI18n } from "../i18n/I18nContext";
import { api } from "../services/api";
import type { EventAlert } from "../types/api";

type EventFilter = "all" | "pending" | "confirmed" | "rejected";
type EventStatus = "pending" | "confirmed" | "rejected";

const getEventStatus = (ev: EventAlert): EventStatus =>
  ev.user_status ?? (ev.acknowledged ? "confirmed" : "pending");

export const EventsPage: React.FC<{ events: EventAlert[]; onRefresh: () => void }> = ({ events, onRefresh }) => {
  const { t } = useI18n();
  const [cfg, setCfg] = useState<Record<string, any> | null>(null);
  const [savedMsg, setSavedMsg] = useState("");
  const [clearMsg, setClearMsg] = useState("");
  const [confirmClearAll, setConfirmClearAll] = useState(false);
  const [filter, setFilter] = useState<EventFilter>("all");
  const [busyId, setBusyId] = useState<number | null>(null);
  const [clearing, setClearing] = useState(false);
  const [previewObsId, setPreviewObsId] = useState<Record<number, boolean>>({});

  useEffect(() => {
    api.getSystemConfig().then((c) => setCfg(c.events)).catch(() => {});
  }, []);

  const setFeedback = async (id: number, status: EventStatus) => {
    setBusyId(id);
    try {
      await api.setEventFeedback(id, status);
      onRefresh();
    } finally {
      setBusyId(null);
    }
  };

  const removeEvent = async (id: number) => {
    setBusyId(id);
    try {
      await api.deleteEvent(id);
      onRefresh();
    } finally {
      setBusyId(null);
    }
  };

  const handleBulkClear = async (scope: "duplicates" | "rejected" | "all") => {
    setClearing(true);
    try {
      const res = await api.clearEvents(scope);
      setConfirmClearAll(false);
      setClearMsg(t("events.clearedMsg", { n: res.deleted ?? 0 }));
      onRefresh();
      window.setTimeout(() => setClearMsg(""), 3000);
    } finally {
      setClearing(false);
    }
  };

  const togglePreview = (id: number) => {
    setPreviewObsId((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const saveThresholds = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!cfg) return;
    await api.updateEventsConfig(cfg);
    setSavedMsg(t("events.savedMsg"));
    setTimeout(() => setSavedMsg(""), 2500);
  };

  const allCount = events.length;
  const pendingCount = events.filter((ev) => getEventStatus(ev) === "pending").length;
  const confirmedCount = events.filter((ev) => getEventStatus(ev) === "confirmed").length;
  const rejectedCount = events.filter((ev) => getEventStatus(ev) === "rejected").length;
  const filteredEvents =
    filter === "all" ? events : events.filter((ev) => getEventStatus(ev) === filter);

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-5 items-start">
      <div className="xl:col-span-2 obs-panel p-5 space-y-4">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-2.5">
            <h2 className="text-base font-semibold text-white flex items-center gap-2">
              <Bell className="w-4 h-4 text-amber-400" /> {t("events.title")}
            </h2>
            <span className="text-xs font-mono tabular-nums text-[#8E95A2]">
              {t("events.count", { n: events.length })}
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            {clearMsg && <span className="text-emerald-400 font-mono mr-1">{clearMsg}</span>}
            {events.length > 0 && (
              <>
                <button
                  type="button"
                  disabled={clearing}
                  onClick={() => handleBulkClear("duplicates")}
                  className="obs-btn-secondary inline-flex items-center gap-1 px-2.5 py-1"
                >
                  <Sparkles className="w-3 h-3 text-sky-400" /> {t("events.dedupBtn")}
                </button>
                {rejectedCount > 0 && (
                  <button
                    type="button"
                    disabled={clearing}
                    onClick={() => handleBulkClear("rejected")}
                    className="obs-btn-secondary px-2.5 py-1 text-rose-300"
                  >
                    {t("events.clearDeniedBtn")}
                  </button>
                )}
                {!confirmClearAll ? (
                  <button
                    type="button"
                    disabled={clearing}
                    onClick={() => setConfirmClearAll(true)}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-rose-500/15 hover:bg-rose-500/25 border border-rose-500/30 text-rose-200"
                  >
                    <Trash2 className="w-3 h-3" /> {t("events.clearAllBtn")}
                  </button>
                ) : (
                  <div className="inline-flex items-center gap-1">
                    <button
                      type="button"
                      disabled={clearing}
                      onClick={() => handleBulkClear("all")}
                      className="px-2.5 py-1 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-semibold"
                    >
                      {t("events.confirmClearAll")}
                    </button>
                    <button
                      type="button"
                      onClick={() => setConfirmClearAll(false)}
                      className="obs-btn-secondary px-2 py-1"
                    >
                      {t("timelapse.cancelBtn")}
                    </button>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        <div className="inline-flex flex-wrap gap-1 bg-[#0B0C0E] border border-white/[0.07] rounded-xl p-1">
          {(
            [
              ["all", t("events.filter.all", { n: allCount })],
              ["pending", t("events.filter.pending", { n: pendingCount })],
              ["confirmed", t("events.filter.confirmed", { n: confirmedCount })],
              ["rejected", t("events.filter.rejected", { n: rejectedCount })],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              type="button"
              onClick={() => setFilter(key)}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition ${
                filter === key
                  ? "bg-[#F4F5F7] text-[#0B0C0E] font-semibold"
                  : "text-[#8E95A2] hover:text-white"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="space-y-2.5 max-h-[600px] overflow-y-auto pr-1">
          {filteredEvents.map((ev) => {
            const status = getEventStatus(ev);
            const isBusy = busyId === ev.id;
            const showFrame = !!previewObsId[ev.id];
            const cardClasses =
              status === "confirmed"
                ? "bg-emerald-950/15 border-emerald-500/25"
                : status === "rejected"
                ? "bg-rose-950/15 border-rose-500/25 opacity-70"
                : "bg-[#1A1D24]/80 border-white/[0.08]";

            return (
              <div key={ev.id} className={`p-3.5 rounded-xl border text-xs space-y-2.5 ${cardClasses}`}>
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase ${
                          ev.severity === "critical"
                            ? "bg-rose-500/20 text-rose-300"
                            : ev.severity === "warning"
                            ? "bg-amber-500/20 text-amber-300"
                            : "bg-sky-500/20 text-sky-300"
                        }`}
                      >
                        {ev.severity}
                      </span>
                      <span
                        className={`font-semibold ${
                          status === "rejected" ? "line-through text-[#8E95A2]" : "text-white"
                        }`}
                      >
                        {ev.title}
                      </span>
                      {status === "confirmed" && (
                        <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 text-[10px] font-medium">
                          {t("events.status.confirmed")}
                        </span>
                      )}
                      {status === "rejected" && (
                        <span className="px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 text-[10px] font-medium">
                          {t("events.status.rejected")}
                        </span>
                      )}
                      <span className="text-[11px] font-mono tabular-nums text-[#646B78]">
                        {new Date(ev.created_at).toLocaleString()}
                      </span>
                    </div>
                    <p className="text-[#C5CBD5] mt-1">{ev.description}</p>
                  </div>

                  <div className="flex items-center gap-1.5 shrink-0 flex-wrap justify-end">
                    {ev.observation_id && (
                      <button
                        type="button"
                        onClick={() => togglePreview(ev.id)}
                        className="obs-btn-secondary inline-flex items-center gap-1 px-2 py-1"
                      >
                        <Image className="w-3 h-3" />
                        {showFrame ? t("events.hideFrame") : t("events.viewFrame")}
                      </button>
                    )}
                    {status !== "confirmed" && (
                      <button
                        type="button"
                        disabled={isBusy}
                        onClick={() => setFeedback(ev.id, "confirmed")}
                        className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-emerald-500/20 hover:bg-emerald-500/35 border border-emerald-500/35 text-emerald-200 disabled:opacity-50"
                      >
                        <Check className="w-3 h-3" /> {t("events.ack")}
                      </button>
                    )}
                    {status !== "rejected" && (
                      <button
                        type="button"
                        disabled={isBusy}
                        onClick={() => setFeedback(ev.id, "rejected")}
                        className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-rose-500/20 hover:bg-rose-500/35 border border-rose-500/35 text-rose-200 disabled:opacity-50"
                      >
                        <X className="w-3 h-3" /> {t("events.deny")}
                      </button>
                    )}
                    {status !== "pending" && (
                      <button
                        type="button"
                        disabled={isBusy}
                        title={t("events.undo")}
                        onClick={() => setFeedback(ev.id, "pending")}
                        className="obs-btn-secondary p-1.5"
                      >
                        <RotateCcw className="w-3 h-3" />
                      </button>
                    )}
                    <button
                      type="button"
                      disabled={isBusy}
                      title={t("events.delete")}
                      onClick={() => removeEvent(ev.id)}
                      className="obs-btn-secondary p-1.5 hover:text-rose-300"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                </div>

                {showFrame && ev.observation_id && (
                  <div className="pt-1">
                    <img
                      src={`/api/analysis/image/${ev.observation_id}?layer=raw`}
                      alt={`Observation #${ev.observation_id}`}
                      className="max-h-48 rounded-xl border border-white/[0.1] object-contain bg-black"
                    />
                  </div>
                )}
              </div>
            );
          })}
          {events.length === 0 && (
            <div className="py-10 text-center text-xs text-[#646B78]">{t("events.empty")}</div>
          )}
          {events.length > 0 && filteredEvents.length === 0 && (
            <div className="py-10 text-center text-xs text-[#646B78]">{t("events.emptyFiltered")}</div>
          )}
        </div>
      </div>

      <form onSubmit={saveThresholds} className="obs-panel p-5 space-y-3.5">
        <h3 className="text-sm font-semibold text-white">{t("events.thresholdsTitle")}</h3>
        {cfg ? (
          <div className="space-y-3 text-xs">
            {(
              [
                ["cooldown_minutes", t("events.cooldownMin"), 1, 1440, 1],
                ["rejected_backoff_minutes", t("events.rejectedBackoffMin"), 5, 1440, 5],
                ["rapid_cloud_increase_pct", t("events.rapidIncrease"), 1, 100, 0.5],
                ["rapid_cloud_decrease_pct", t("events.rapidDecrease"), 1, 100, 0.5],
                ["rapid_darkening_delta", t("events.rapidDarkening"), -100, -1, 0.5],
                ["dense_cloud_alert_pct", t("events.denseCloud"), 5, 100, 1],
                ["storm_like_dark_cloud_pct", t("events.stormDark"), 5, 100, 1],
                ["camera_move_shift_px", t("events.camMove"), 2, 500, 1],
                ["min_storage_free_gb", t("events.lowStorage"), 0.5, 500, 0.5],
              ] as const
            ).map(([key, label, min, max, step]) => (
              <label key={key} className="block space-y-1">
                <span className="text-[#9AA1AD]">{label}</span>
                <input
                  type="number"
                  step={step}
                  min={min}
                  max={max}
                  value={cfg[key] ?? 0}
                  onChange={(e) => setCfg({ ...cfg, [key]: Number(e.target.value) })}
                  className="obs-input w-full font-mono tabular-nums"
                />
              </label>
            ))}
            <label className="flex items-center gap-2 pt-1 text-[#C5CBD5]">
              <input
                type="checkbox"
                checked={cfg.camera_movement_enabled ?? true}
                onChange={(e) => setCfg({ ...cfg, camera_movement_enabled: e.target.checked })}
              />
              {t("events.camMoveEnabled")}
            </label>
            <label className="flex items-center gap-2 text-[#C5CBD5]">
              <input
                type="checkbox"
                checked={!!cfg.browser_notifications_enabled}
                onChange={(e) => setCfg({ ...cfg, browser_notifications_enabled: e.target.checked })}
              />
              {t("events.browserNotif")}
            </label>
            <button type="submit" className="obs-btn-primary w-full inline-flex items-center justify-center gap-1.5 px-3 py-2">
              <Save className="w-3.5 h-3.5" /> {t("events.saveThresholds")}
            </button>
            {savedMsg && <p className="text-emerald-400 text-center">{savedMsg}</p>}
          </div>
        ) : (
          <p className="text-xs text-[#646B78]">{t("events.loading")}</p>
        )}
      </form>
    </div>
  );
};
