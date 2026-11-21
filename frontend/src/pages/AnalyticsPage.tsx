import React, { useState } from "react";
import { useI18n } from "../i18n/I18nContext";
import type { ForecastItem } from "../types/api";
import { ForecastPage } from "./ForecastPage";
import { TrendsPage } from "./TrendsPage";

export const AnalyticsPage: React.FC<{ forecasts: ForecastItem[] }> = ({ forecasts }) => {
  const { t } = useI18n();
  const [tab, setTab] = useState<"nowcast" | "trends">("nowcast");

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="inline-flex bg-[#13151A] border border-white/[0.07] rounded-xl p-1 gap-1">
          <button
            type="button"
            onClick={() => setTab("nowcast")}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition ${
              tab === "nowcast" ? "bg-[#F4F5F7] text-[#0B0C0E] font-semibold" : "text-[#9AA1AD] hover:text-white"
            }`}
          >
            {t("nav.sub.nowcast")}
          </button>
          <button
            type="button"
            onClick={() => setTab("trends")}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition ${
              tab === "trends" ? "bg-[#F4F5F7] text-[#0B0C0E] font-semibold" : "text-[#9AA1AD] hover:text-white"
            }`}
          >
            {t("nav.sub.trends")}
          </button>
        </div>
      </div>

      {tab === "nowcast" ? <ForecastPage forecasts={forecasts} /> : <TrendsPage />}
    </div>
  );
};
