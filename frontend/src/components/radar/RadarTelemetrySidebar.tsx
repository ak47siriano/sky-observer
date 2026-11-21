import React from "react";
import { useI18n } from "../../i18n/I18nContext";

interface RadarTelemetrySidebarProps {
  obs: any;
  state: any;
  wx: any;
  isLive: boolean;
  activeTimeSec: number | null;
  offlineMode: boolean;
  onToggleMode: (offline: boolean) => void;
}

export const RadarTelemetrySidebar: React.FC<RadarTelemetrySidebarProps> = ({
  obs,
  state,
  wx,
  isLive,
  activeTimeSec,
  offlineMode,
  onToggleMode,
}) => {
  const { t, tEnum, tCardinal } = useI18n();
  const altM = Math.round(Number(state?.camera_projection?.cloud_altitude_m || 2000));
  const altSource = state?.camera_projection?.cloud_altitude_source;

  return (
    <div className="obs-panel divide-y divide-[#262B34] text-xs">
      {/* Section 1: Local Camera Observation */}
      <div className="p-3.5 space-y-2.5">
        <div className="flex items-center justify-between gap-2">
          <span className="obs-label">{t("radar.localCamCard")}</span>
          {obs && (
            <span className="font-mono tabular-nums text-[10px] text-[#8D96A5]">
              {isLive
                ? "LIVE"
                : activeTimeSec != null
                ? new Date(activeTimeSec * 1000).toLocaleTimeString()
                : t("radar.timelineHistory")}{" "}
              · #{obs.id}
            </span>
          )}
        </div>

        {obs ? (
          <div className="divide-y divide-[#262B34]/70">
            <div className="py-1.5 flex justify-between items-center">
              <span className="text-[#8D96A5]">{t("overview.cloudCoverage")}</span>
              <span className="obs-metric-value font-semibold">
                {obs.cloud_coverage_pct.toFixed(1)}% ({Math.round((obs.cloud_coverage_pct / 100) * 8)}/8{" "}
                {obs.cloud_coverage_pct < 6
                  ? "SKC"
                  : obs.cloud_coverage_pct < 31
                  ? "FEW"
                  : obs.cloud_coverage_pct < 56
                  ? "SCT"
                  : obs.cloud_coverage_pct < 88
                  ? "BKN"
                  : "OVC"}
                )
              </span>
            </div>
            <div className="py-1.5 flex justify-between items-center">
              <span className="text-[#8D96A5]">{t("overview.dominantCloudType")}</span>
              <span className="text-[#ECEFF4] font-medium">{tEnum(obs.dominant_cloud_type)}</span>
            </div>
            <div className="py-1.5 flex justify-between items-center">
              <span className="text-[#8D96A5]">{t("overview.cloudAdvection")}</span>
              <span className="obs-metric-value">
                {obs.motion_speed_px_s.toFixed(1)} px/s {tCardinal(obs.motion_direction_cardinal)}
              </span>
            </div>
            {obs.motion?.secondary_layer?.available && (
              <div className="py-1.5 flex justify-between items-center text-[11px] font-mono tabular-nums">
                <span className="text-[#8D96A5]">2nd Deck ({obs.motion.secondary_layer.share_pct}%)</span>
                <span className="text-[#ECEFF4]">
                  {Number(obs.motion.secondary_layer.magnitude_px_s).toFixed(1)} px/s{" "}
                  {tCardinal(obs.motion.secondary_layer.direction_cardinal)}
                </span>
              </div>
            )}
            {obs.motion_speed_px_s > 0.2 && (
              <div className="py-1.5 flex justify-between items-center text-[11px] font-mono tabular-nums">
                <span className="text-[#8D96A5]">Ground Speed</span>
                <span className="text-[#ECEFF4]">
                  {t("radar.derivedSpeed", {
                    kmh: (
                      ((Number(obs.motion_speed_deg_s ?? (obs.motion_speed_px_s * 72) / 1920) * Math.PI) / 180) *
                      Number(state?.camera_projection?.cloud_altitude_m || 2000) *
                      3.6
                    ).toFixed(1),
                    alt: altM,
                  })}
                </span>
              </div>
            )}
            <div className="py-1.5 flex justify-between items-center text-[11px] font-mono tabular-nums text-[#8D96A5]">
              <span>
                Tilt {Math.round(90 - Number(state?.location?.orientation_elevation_deg ?? 65))}° · Az{" "}
                {Math.round(Number(state?.location?.orientation_azimuth_deg ?? 0))}°
              </span>
              {altSource && <span>Base ~{altM} m</span>}
            </div>
            <div className="pt-2 grid grid-cols-3 divide-x divide-[#262B34] text-center font-mono tabular-nums text-[11px]">
              <div className="px-1">
                <div className="text-[#8D96A5]">{t("overview.thin")}</div>
                <div className="text-[#ECEFF4] font-semibold mt-0.5">{obs.thin_cloud_pct.toFixed(0)}%</div>
              </div>
              <div className="px-1">
                <div className="text-[#8D96A5]">{t("overview.dense")}</div>
                <div className="text-[#ECEFF4] font-semibold mt-0.5">{obs.dense_cloud_pct.toFixed(0)}%</div>
              </div>
              <div className="px-1">
                <div className="text-[#8D96A5]">{t("overview.dark")}</div>
                <div className="text-[#ECEFF4] font-semibold mt-0.5">{obs.dark_cloud_pct.toFixed(0)}%</div>
              </div>
            </div>
          </div>
        ) : (
          <p className="text-[#5D6675]">{t("overview.noImageYet")}</p>
        )}
      </div>

      {/* Section 2: Open-Meteo Surface & NWP Table */}
      <div className="p-3.5 space-y-2.5">
        <div className="flex items-center justify-between gap-2">
          <span className="obs-label">{t("radar.onlineCard")}</span>
          <span className="text-[10px] font-mono text-[#8D96A5]">OPEN-METEO</span>
        </div>
        {offlineMode ? (
          <div className="space-y-2 text-[#8D96A5]">
            <p>{t("radar.offlineExplain")}</p>
            <button
              type="button"
              onClick={() => onToggleMode(false)}
              className="obs-btn-primary w-full px-3 py-1.5"
            >
              {t("radar.enableOnlineBtn")}
            </button>
          </div>
        ) : wx ? (
          <div>
            <div className="pb-2 mb-2 border-b border-[#262B34] flex items-center justify-between gap-2">
              <span className="font-semibold text-[#ECEFF4] truncate">
                {state?.online_weather?.location_name ||
                  wx.location_name ||
                  state?.location?.location_description ||
                  `${Number(wx.latitude ?? state?.location?.latitude ?? 0).toFixed(3)}°, ${Number(
                    wx.longitude ?? state?.location?.longitude ?? 0
                  ).toFixed(3)}°`}
              </span>
              <span className="text-[10px] font-mono tabular-nums text-[#8D96A5] shrink-0">
                {Number(wx.latitude ?? state?.location?.latitude ?? 0).toFixed(2)}°,{" "}
                {Number(wx.longitude ?? state?.location?.longitude ?? 0).toFixed(2)}°
              </span>
            </div>
            <div className="grid grid-cols-2 divide-x divide-y divide-[#262B34] border border-[#262B34] rounded">
              <div className="p-2">
                <div className="text-[10px] text-[#8D96A5]">{t("wx.temp")}</div>
                <div className="obs-metric-value text-sm font-semibold">{wx.temperature_c}°C</div>
              </div>
              <div className="p-2">
                <div className="text-[10px] text-[#8D96A5]">{t("wx.humidity")}</div>
                <div className="obs-metric-value text-sm font-semibold">{wx.relative_humidity_pct}%</div>
              </div>
              <div className="p-2">
                <div className="text-[10px] text-[#8D96A5]">{t("wx.pressure")}</div>
                <div className="obs-metric-value text-sm font-semibold">{wx.pressure_msl_hpa} hPa</div>
              </div>
              <div className="p-2">
                <div className="text-[10px] text-[#8D96A5]">{t("wx.wind")}</div>
                <div className="obs-metric-value text-sm font-semibold">{wx.wind_speed_kmh} km/h</div>
              </div>
              <div className="p-2">
                <div className="text-[10px] text-[#8D96A5]">{t("wx.apiClouds")}</div>
                <div className="obs-metric-value text-sm font-semibold">{wx.cloud_cover_pct}%</div>
              </div>
              <div className="p-2">
                <div className="text-[10px] text-[#8D96A5]">{t("wx.rain")}</div>
                <div className="obs-metric-value text-sm font-semibold">{wx.rain_mm} mm</div>
              </div>
              {wx.dewpoint_c != null && (
                <div className="p-2">
                  <div className="text-[10px] text-[#8D96A5]">{t("radar.dewpoint")}</div>
                  <div className="obs-metric-value text-sm font-semibold">
                    {wx.dewpoint_c}°C
                    {wx.temperature_c != null && (
                      <span className="text-[10px] font-normal text-[#8D96A5] ml-1">
                        (Δ{(Number(wx.temperature_c) - Number(wx.dewpoint_c)).toFixed(1)}°)
                      </span>
                    )}
                  </div>
                </div>
              )}
              {wx.estimated_cloud_base_m != null && (
                <div className="p-2">
                  <div className="text-[10px] text-[#8D96A5]">{t("radar.cloudBaseLcl")}</div>
                  <div className="obs-metric-value text-sm font-semibold">~{wx.estimated_cloud_base_m} m</div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <p className="text-[#8D96A5]">{t("radar.fetchingOnline")}</p>
        )}
      </div>
    </div>
  );
};
