import React, { useEffect, useMemo, useState } from "react";
import {
  Bell,
  BookOpen,
  Calculator,
  CheckCircle2,
  CloudSun,
  Cpu,
  LayoutDashboard,
  Lightbulb,
  MapPin,
  Search,
  Settings,
  X,
} from "lucide-react";
import { useI18n } from "../i18n/I18nContext";
import {
  MANUAL_SECTIONS_EN,
  MANUAL_SECTIONS_IT,
  type ManualSectionId,
} from "../data/manualContent";

interface ManualModalProps {
  open: boolean;
  currentPage: string;
  onClose: () => void;
}

const SECTION_ICONS: Record<ManualSectionId, React.ComponentType<{ className?: string }>> = {
  dashboard: LayoutDashboard,
  radar: MapPin,
  forecast: CloudSun,
  events: Bell,
  ai: Cpu,
  settings: Settings,
  formulas: Calculator,
};

function pageToSectionId(page: string): ManualSectionId {
  if (["dashboard", "radar", "forecast", "events", "ai", "settings"].includes(page)) {
    return page as ManualSectionId;
  }
  return "dashboard";
}

export const ManualModal: React.FC<ManualModalProps> = ({ open, currentPage, onClose }) => {
  const { locale } = useI18n();
  const sections = locale === "it" ? MANUAL_SECTIONS_IT : MANUAL_SECTIONS_EN;
  const contextSectionId = pageToSectionId(currentPage);

  const [activeId, setActiveId] = useState<ManualSectionId>(contextSectionId);
  const [query, setQuery] = useState("");

  // Whenever the modal opens, jump directly to the section matching the current page
  useEffect(() => {
    if (open) {
      setActiveId(pageToSectionId(currentPage));
      setQuery("");
    }
  }, [open, currentPage]);

  // Close on Escape key
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const trimmedQuery = query.trim().toLowerCase();

  const searchResults = useMemo(() => {
    if (!trimmedQuery) return [];
    const hits: { sectionId: ManualSectionId; sectionTitle: string; param: (typeof sections)[0]["params"][0] }[] = [];
    for (const sec of sections) {
      for (const p of sec.params) {
        const hay = `${sec.title} ${p.name} ${p.badge || ""} ${p.what} ${p.how} ${p.tip || ""}`.toLowerCase();
        if (hay.includes(trimmedQuery)) {
          hits.push({ sectionId: sec.id, sectionTitle: sec.shortTitle, param: p });
        }
      }
    }
    return hits;
  }, [sections, trimmedQuery]);

  if (!open) return null;

  const activeSection = sections.find((s) => s.id === activeId) || sections[0];
  const isIt = locale === "it";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6 bg-black/70 backdrop-blur-md"
      onClick={onClose}
    >
      <div
        className="obs-panel w-full max-w-5xl h-[86vh] flex flex-col overflow-hidden border-sky-400/25 shadow-[0_24px_90px_rgba(0,0,0,0.85)]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Top Modal Header */}
        <div className="px-5 py-3.5 border-b border-white/[0.08] bg-[#090D15]/90 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-sky-400/15 border border-sky-400/30 flex items-center justify-center text-sky-300">
              <BookOpen className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-white tracking-tight">
                {isIt ? "Manuale Completo di Sky Observer" : "Sky Observer Complete User Manual"}
              </h2>
              <p className="text-[11px] text-[#8A96AA]">
                {isIt
                  ? "Guida semplice all'utilizzo e spiegazione chiara di come viene calcolato ogni parametro."
                  : "Easy-to-read usage guide and clear explanation of how every parameter is calculated."}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 flex-1 sm:flex-initial justify-end">
            <div className="relative w-full sm:w-64">
              <Search className="w-3.5 h-3.5 text-[#64748B] absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={isIt ? "Cerca parametro (es. LCL, Tilt, MAE)..." : "Search parameter (e.g. LCL, Tilt, MAE)..."}
                className="obs-input w-full pl-8 py-1.5 text-xs"
              />
            </div>
            <button
              type="button"
              onClick={onClose}
              title="Close (Esc)"
              className="obs-btn-secondary p-1.5 text-slate-300 hover:text-white"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Body: Left Indexed TOC + Right Content */}
        <div className="flex-1 min-h-0 flex flex-col md:flex-row">
          {/* Indexed Sidebar TOC */}
          <aside className="md:w-60 shrink-0 border-b md:border-b-0 md:border-r border-white/[0.08] bg-[#0B0F18]/75 p-2.5 overflow-x-auto md:overflow-y-auto flex md:flex-col gap-1 no-scrollbar">
            <div className="hidden md:block px-2.5 pb-1.5 pt-1 obs-label">
              {isIt ? "Indice del Manuale" : "Manual Index"}
            </div>
            {sections.map((sec) => {
              const Icon = SECTION_ICONS[sec.id];
              const isActive = !trimmedQuery && activeId === sec.id;
              const isCurrentPage = sec.id === contextSectionId;
              return (
                <button
                  key={sec.id}
                  type="button"
                  onClick={() => {
                    setQuery("");
                    setActiveId(sec.id);
                  }}
                  className={`flex items-center justify-between gap-2 px-2.5 py-2 rounded-xl text-left text-xs font-medium whitespace-nowrap md:whitespace-normal transition-all shrink-0 ${
                    isActive
                      ? "bg-sky-400/15 border border-sky-400/35 text-sky-200 shadow-[0_2px_12px_rgba(56,189,248,0.14)]"
                      : "text-[#94A3B8] hover:text-white hover:bg-white/[0.05] border border-transparent"
                  }`}
                >
                  <span className="flex items-center gap-2 min-w-0">
                    <Icon className={`w-3.5 h-3.5 shrink-0 ${isActive ? "text-sky-300" : ""}`} />
                    <span className="truncate">{sec.shortTitle}</span>
                  </span>
                  {isCurrentPage && (
                    <span className="hidden md:inline-block shrink-0 px-1.5 py-0.5 rounded-md bg-emerald-500/15 border border-emerald-400/30 text-emerald-300 text-[9px] font-mono">
                      {isIt ? "QUI" : "HERE"}
                    </span>
                  )}
                </button>
              );
            })}
          </aside>

          {/* Main Reading Pane */}
          <div className="flex-1 overflow-y-auto p-5 sm:p-6 space-y-5">
            {trimmedQuery ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-white">
                    {isIt
                      ? `Risultati per "${query}" (${searchResults.length})`
                      : `Results for "${query}" (${searchResults.length})`}
                  </h3>
                  <button
                    type="button"
                    onClick={() => setQuery("")}
                    className="text-xs text-sky-300 hover:underline"
                  >
                    {isIt ? "Torna all'indice" : "Back to index"}
                  </button>
                </div>
                {searchResults.length === 0 ? (
                  <div className="obs-inset p-6 text-center text-xs text-[#8A96AA]">
                    {isIt
                      ? "Nessun parametro trovato. Prova con: Copertura, Tilt, LCL, MAE, mIoU, DirectML, Timelapse."
                      : "No matching parameter found. Try: Coverage, Tilt, LCL, MAE, mIoU, DirectML, Timelapse."}
                  </div>
                ) : (
                  <div className="space-y-3">
                    {searchResults.map((hit, idx) => (
                      <div key={idx} className="obs-inset p-4 space-y-2 text-xs">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <span className="text-sm font-semibold text-white">{hit.param.name}</span>
                          <div className="flex items-center gap-1.5">
                            {hit.param.badge && (
                              <span className="px-2 py-0.5 rounded-md bg-sky-500/15 border border-sky-400/30 text-sky-200 font-mono text-[10px]">
                                {hit.param.badge}
                              </span>
                            )}
                            <button
                              type="button"
                              onClick={() => {
                                setQuery("");
                                setActiveId(hit.sectionId);
                              }}
                              className="px-2 py-0.5 rounded-md bg-white/[0.06] text-[#94A3B8] hover:text-white text-[10px]"
                            >
                              {hit.sectionTitle} →
                            </button>
                          </div>
                        </div>
                        <p className="text-[#CBD5E1] leading-relaxed">
                          <strong className="text-white">{isIt ? "Cos'è: " : "What it is: "}</strong>
                          {hit.param.what}
                        </p>
                        <p className="text-[#94A3B8] leading-relaxed">
                          <strong className="text-sky-300">{isIt ? "Come si calcola: " : "How it's calculated: "}</strong>
                          {hit.param.how}
                        </p>
                        {hit.param.tip && (
                          <div className="flex items-start gap-2 pt-1.5 text-[11px] text-emerald-200/90">
                            <Lightbulb className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                            <span>{hit.param.tip}</span>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <>
                {/* Chapter Header */}
                <div className="space-y-1.5 border-b border-white/[0.07] pb-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="text-lg font-bold text-white tracking-tight">{activeSection.title}</h3>
                    {activeSection.id === contextSectionId && (
                      <span className="px-2 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-400/30 text-emerald-300 text-[10px] font-mono">
                        {isIt ? "SEZIONE PAGINA ATTUALE" : "CURRENT PAGE"}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-sky-300/90">{activeSection.subtitle}</p>
                  <p className="text-xs text-[#CBD5E1] leading-relaxed pt-1">{activeSection.intro}</p>
                </div>

                {/* Quick Steps */}
                {activeSection.quickSteps && activeSection.quickSteps.length > 0 && (
                  <div className="obs-inset p-4 space-y-2 border-sky-400/20 bg-sky-500/[0.05]">
                    <div className="text-xs font-semibold text-sky-200 flex items-center gap-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5 text-sky-400" />
                      {isIt ? "Come usarla in pratica (passi rapidi)" : "Quick Start Steps"}
                    </div>
                    <ol className="space-y-1.5 text-xs text-[#D6DFEC] list-decimal list-inside">
                      {activeSection.quickSteps.map((step, i) => (
                        <li key={i} className="leading-relaxed">
                          {step}
                        </li>
                      ))}
                    </ol>
                  </div>
                )}

                {/* Parameter Breakdown Cards */}
                <div className="space-y-3">
                  <div className="obs-label">
                    {isIt ? "Parametri & Come Vengono Calcolati" : "Parameters & How They Are Calculated"}
                  </div>
                  {activeSection.params.map((p, idx) => (
                    <div key={idx} className="obs-inset p-4 space-y-2 text-xs hover:border-white/[0.14] transition">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <span className="text-[13px] font-semibold text-white">{p.name}</span>
                        {p.badge && (
                          <span className="px-2 py-0.5 rounded-md bg-sky-500/15 border border-sky-400/30 text-sky-200 font-mono text-[10px]">
                            {p.badge}
                          </span>
                        )}
                      </div>
                      <p className="text-[#E2E8F0] leading-relaxed">
                        <strong className="text-white">{isIt ? "Cosa significa: " : "What it means: "}</strong>
                        {p.what}
                      </p>
                      <div className="rounded-lg bg-[#0A0E16]/85 border border-white/[0.06] px-3 py-2 text-[#B6C2D4] leading-relaxed">
                        <strong className="text-sky-300">{isIt ? "Come viene calcolato: " : "How it's calculated: "}</strong>
                        {p.how}
                      </div>
                      {p.tip && (
                        <div className="flex items-start gap-2 pt-0.5 text-[11px] text-emerald-200/90">
                          <Lightbulb className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                          <span>
                            <strong>{isIt ? "Consiglio: " : "Tip: "}</strong>
                            {p.tip}
                          </span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
