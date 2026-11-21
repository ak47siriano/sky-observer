import React, { useState } from "react";
import { useI18n } from "../i18n/I18nContext";
import { DatasetLabelingPage } from "./DatasetLabelingPage";
import { ModelLabPage } from "./ModelLabPage";

export const AiStudioPage: React.FC = () => {
  const { t } = useI18n();
  const [tab, setTab] = useState<"models" | "labeling">("models");

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="inline-flex bg-[#13151A] border border-white/[0.07] rounded-xl p-1 gap-1">
          <button
            type="button"
            onClick={() => setTab("models")}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition ${
              tab === "models" ? "bg-[#F4F5F7] text-[#0B0C0E] font-semibold" : "text-[#9AA1AD] hover:text-white"
            }`}
          >
            {t("nav.sub.models")}
          </button>
          <button
            type="button"
            onClick={() => setTab("labeling")}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition ${
              tab === "labeling" ? "bg-[#F4F5F7] text-[#0B0C0E] font-semibold" : "text-[#9AA1AD] hover:text-white"
            }`}
          >
            {t("nav.sub.labeling")}
          </button>
        </div>
      </div>

      {tab === "models" ? <ModelLabPage /> : <DatasetLabelingPage />}
    </div>
  );
};
