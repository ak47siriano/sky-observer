async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(init?.headers || {}),
    },
    ...init,
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const data = await res.json();
      msg = data.detail || data.message || msg;
    } catch {
      // ignore
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  getSystemStatus: () => request<any>("/api/system/status"),
  getSystemConfig: () => request<any>("/api/system/config"),
  updateSystemConfig: (patch: Record<string, any>) =>
    request<any>("/api/system/config", { method: "PUT", body: JSON.stringify(patch) }),
  updateHardwareProfile: (payload: {
    profile?: "auto" | "low_end" | "balanced" | "high_end" | "ultra" | "custom";
    device?: "auto" | "cuda" | "rocm" | "directml" | "mps" | "cpu";
    inference_resolution?: number;
    batch_size?: number;
    cpu_threads?: number;
    max_gpu_memory_fraction?: number;
    mixed_precision?: boolean;
  }) => request<any>("/api/system/hardware-profile", { method: "PUT", body: JSON.stringify(payload) }),
  getStorageStats: () => request<any>("/api/system/storage"),
  cleanupStorage: () => request<any>("/api/system/storage/cleanup", { method: "POST" }),
  purgeStorage: (payload: {
    targets: string[];
    preserve_training_samples?: boolean;
    preserve_camera_connection?: boolean;
  }) => request<any>("/api/system/storage/purge", { method: "POST", body: JSON.stringify(payload) }),
  deleteObservation: (id: number) =>
    request<any>(`/api/analysis/observation/${id}`, { method: "DELETE" }),

  getCameraStatus: () => request<any>("/api/camera/status"),
  getCameraConfig: () => request<any>("/api/camera/config"),
  updateCameraConfig: (cfg: Record<string, any>) =>
    request<any>("/api/camera/config", { method: "PUT", body: JSON.stringify(cfg) }),
  updateCameraGeometry: (payload: {
    orientation_elevation_deg: number;
    orientation_azimuth_deg: number;
    orientation_roll_deg?: number;
    fov_horizontal_deg?: number | null;
    fov_vertical_deg?: number | null;
    assumed_cloud_altitude_m?: number | null;
    tilt_correction_enabled?: boolean;
    recalculate_latest?: boolean;
  }) => request<any>("/api/camera/geometry", { method: "PUT", body: JSON.stringify(payload) }),
  autoDetectCameraTilt: (payload?: {
    apply?: boolean;
    fov_horizontal_deg?: number | null;
    fov_vertical_deg?: number | null;
    recalculate_latest?: boolean;
  }) => request<any>("/api/camera/geometry/auto-detect", { method: "POST", body: JSON.stringify(payload || {}) }),
  updateMasks: (payload: { sky_roi_polygon: number[][]; exclusion_polygons: any[] }) =>
    request<any>("/api/camera/masks", { method: "PUT", body: JSON.stringify(payload) }),
  savePaintedExclusionMask: (maskPngBase64: string) =>
    request<any>("/api/camera/exclusion-mask", {
      method: "PUT",
      body: JSON.stringify({ mask_png_base64: maskPngBase64 }),
    }),
  clearPaintedExclusionMask: () => request<any>("/api/camera/exclusion-mask", { method: "DELETE" }),
  proposeExclusionMask: (observationId?: number) =>
    request<any>(
      `/api/camera/exclusion-mask/propose${observationId != null ? `?observation_id=${observationId}` : ""}`,
      { method: "POST" }
    ),
  testCamera: (payload?: { source_type?: string; source_url?: string; device_index?: number }) =>
    request<any>("/api/camera/test", { method: "POST", body: JSON.stringify(payload || {}) }),
  listCameraDevices: () => request<any>("/api/camera/devices"),
  triggerSnapshot: () => request<any>("/api/camera/snapshot"),
  uploadCameraFile: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<any>("/api/camera/upload", { method: "POST", body: fd });
  },

  getLatestAnalysis: () => request<any>("/api/analysis/latest"),
  getAnalysisHistory: (limit = 120, hours?: number) =>
    request<any>(`/api/analysis/history?limit=${limit}${hours ? `&hours=${hours}` : ""}`),

  getLatestForecast: () => request<any>("/api/forecast/latest"),
  getForecastHistory: (limit = 60) => request<any>(`/api/forecast/history?limit=${limit}`),
  getForecastVerification: () => request<any>("/api/forecast/verification"),

  getEvents: (limit = 100) => request<any>(`/api/events?limit=${limit}`),
  ackEvent: (id: number) => request<any>(`/api/events/${id}/ack`, { method: "POST" }),
  rejectEvent: (id: number) => request<any>(`/api/events/${id}/reject`, { method: "POST" }),
  setEventFeedback: (id: number, status: "pending" | "confirmed" | "rejected", note = "") =>
    request<any>(`/api/events/${id}/feedback`, {
      method: "POST",
      body: JSON.stringify({ status, note }),
    }),
  deleteEvent: (id: number) => request<any>(`/api/events/${id}`, { method: "DELETE" }),
  clearEvents: (scope: "pending" | "rejected" | "all" | "duplicates") =>
    request<any>("/api/events/clear", { method: "POST", body: JSON.stringify({ scope }) }),
  updateEventsConfig: (cfg: Record<string, any>) =>
    request<any>("/api/events/config", { method: "PUT", body: JSON.stringify(cfg) }),

  getModels: () => request<any>("/api/models"),
  getActiveModels: () => request<any>("/api/models/active"),
  startTraining: (payload: Record<string, any>) =>
    request<any>("/api/models/train", { method: "POST", body: JSON.stringify(payload) }),
  getTrainingStatus: () => request<any>("/api/models/training/status"),
  stopTraining: (jobId?: string) =>
    request<any>(`/api/models/training/stop${jobId ? `?job_id=${encodeURIComponent(jobId)}` : ""}`, { method: "POST" }),
  activateModel: (modelId: string) =>
    request<any>("/api/models/activate", { method: "POST", body: JSON.stringify({ model_id: modelId }) }),
  deactivateModelFamily: (modelFamily: string) =>
    request<any>("/api/models/deactivate", { method: "POST", body: JSON.stringify({ model_family: modelFamily }) }),
  rollbackModel: (modelFamily: string) =>
    request<any>("/api/models/rollback", { method: "POST", body: JSON.stringify({ model_family: modelFamily }) }),
  deleteModel: (modelId: string, force = false) =>
    request<any>(`/api/models/${encodeURIComponent(modelId)}${force ? "?force=true" : ""}`, { method: "DELETE" }),
  pruneModels: (payload?: {
    keep_per_family?: number;
    model_family?: string | null;
    also_prune_sessions?: boolean;
  }) => request<any>("/api/models/prune", { method: "POST", body: JSON.stringify(payload || {}) }),
  clearTrainingSessions: () => request<any>("/api/models/training/sessions", { method: "DELETE" }),
  getAutoTrainStatus: () => request<any>("/api/models/auto-train/status"),
  updateAutoTrainConfig: (patch: Record<string, any>) =>
    request<any>("/api/models/auto-train/config", { method: "PUT", body: JSON.stringify(patch) }),
  runAutoTrainNow: (payload?: { families?: string[]; epochs?: number; only_label?: boolean }) =>
    request<any>("/api/models/auto-train/run", { method: "POST", body: JSON.stringify(payload || {}) }),
  stopAutoTrain: () => request<any>("/api/models/auto-train/stop", { method: "POST" }),

  getDatasets: () => request<any>("/api/dataset"),
  bootstrapDataset: () => request<any>("/api/dataset/bootstrap", { method: "POST" }),
  importDataset: (payload: Record<string, any>) =>
    request<any>("/api/dataset/import", { method: "POST", body: JSON.stringify(payload) }),
  saveLabel: (payload: Record<string, any>) =>
    request<any>("/api/dataset/label", { method: "POST", body: JSON.stringify(payload) }),
  exportDataset: (datasetId: number) =>
    request<any>("/api/dataset/export", { method: "POST", body: JSON.stringify({ dataset_id: datasetId }) }),

  getTimelapseFrames: (date?: string, limit = 300) =>
    request<any>(`/api/timelapse/frames?limit=${limit}${date ? `&date=${encodeURIComponent(date)}` : ""}`),
  exportTimelapseUrl: "/api/timelapse/export",
  getTimelapseDigests: () => request<any>("/api/timelapse/digests"),
  generateDailyDigest: (payload?: { date?: string | null; layer?: string; fps?: number; include_hud?: boolean }) =>
    request<any>("/api/timelapse/daily-digest", { method: "POST", body: JSON.stringify(payload || {}) }),
  clearTimelapse: (payload?: { date?: string | null; delete_files?: boolean }) =>
    request<any>("/api/timelapse/clear", {
      method: "POST",
      body: JSON.stringify(payload || {}),
    }),
  exportAnalyticsUrl: (format: "csv" | "json", hours?: number) =>
    `/api/analytics/export?format=${format}${hours ? `&hours=${hours}` : ""}`,

  getSecondaryCameras: () => request<any>("/api/cameras/secondary"),
  updateSecondaryCameras: (cameras: any[]) =>
    request<any>("/api/cameras/secondary", { method: "PUT", body: JSON.stringify({ cameras }) }),
  snapshotSecondaryCamera: (camId: string) =>
    request<any>(`/api/cameras/secondary/${encodeURIComponent(camId)}/snapshot`, { method: "POST" }),

  setSystemMode: (offlineMode: boolean) =>
    request<any>("/api/system/mode", { method: "PUT", body: JSON.stringify({ offline_mode: offlineMode }) }),
  getRadarState: () => request<any>("/api/radar/state"),
  updateRadarLocation: (payload: {
    latitude: number;
    longitude: number;
    location_description?: string;
    radar_radius_km?: number;
  }) => request<any>("/api/radar/location", { method: "PUT", body: JSON.stringify(payload) }),
  getOnlineWeather: () => request<any>("/api/weather/online"),
  prefetchOfflineMapTiles: (payload?: { latitude?: number; longitude?: number; styles?: string[] }) =>
    request<any>("/api/radar/tiles/prefetch", { method: "POST", body: JSON.stringify(payload || {}) }),
  searchLocation: (q: string, lang = "en") =>
    request<any>(`/api/weather/geocode?q=${encodeURIComponent(q)}&lang=${encodeURIComponent(lang)}`),
};
