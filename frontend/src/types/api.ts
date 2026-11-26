export interface CameraStatus {
  connected: boolean;
  source_type: string;
  source_url: string;
  device_index?: number;
  camera_name: string;
  location_description: string;
  is_demo_mode: boolean;
  last_capture_at: string | null;
  last_latency_ms: number;
  last_error: string;
  consecutive_failures: number;
  reconnect_attempts: number;
}

export interface Observation {
  id: number;
  captured_at: string;
  source_type: string;
  width: number;
  height: number;
  quality_score: number;
  is_usable: boolean;
  quality: Record<string, any>;
  cloud_coverage_pct: number;
  clear_sky_pct: number;
  thin_cloud_pct: number;
  dense_cloud_pct: number;
  dark_cloud_pct: number;
  sun_glare_pct?: number;
  haze_fog_pct: number;
  uncertainty_pct: number;
  coverage: {
    trend?: string;
    rate_of_change_pct_per_min?: number;
    confidence?: number;
    grid?: Array<{ row: number; col: number; valid_pixels: number; cloud_pct: number | null }>;
    [key: string]: any;
  };
  dominant_cloud_type: string;
  classification_confidence: number;
  classification_probabilities: Record<string, number>;
  motion: Record<string, any>;
  motion_speed_px_s: number;
  motion_speed_deg_s: number | null;
  motion_direction_deg: number;
  motion_direction_cardinal: string;
  motion_consistency: number;
  motion_confidence: number;
  avg_luminance: number;
  blue_sky_index: number;
  color: Record<string, any>;
  texture: Record<string, any>;
  visual_condition: string;
  inference_duration_ms: number;
  capture_latency_ms: number;
  has_raw: boolean;
  has_seg_overlay: boolean;
  has_motion_overlay: boolean;
  is_training_sample: boolean;
}

export interface ForecastItem {
  id: number;
  generated_at: string;
  observation_id: number | null;
  horizon_minutes: number;
  target_time: string;
  status: string;
  forecast: string;
  cloud_cover_prediction: number;
  clearing_prob: number;
  increasing_prob: number;
  overcast_prob: number;
  darkening_prob: number;
  precip_signal_prob: number;
  storm_signal_prob: number;
  confidence: number;
  signals: string[];
  reasons: string[];
  limitations: string[];
  baselines?: Record<string, any>;
  low_reliability?: boolean;
  cover_p10?: number | null;
  cover_p90?: number | null;
  model_version: string;
  verified: boolean;
}

export interface EventAlert {
  id: number;
  created_at: string;
  event_type: string;
  severity: "info" | "warning" | "critical";
  title: string;
  description: string;
  observation_id: number | null;
  metrics: Record<string, any>;
  acknowledged: boolean;
  user_status?: "pending" | "confirmed" | "rejected";
  user_note?: string;
}

export interface ModelEntry {
  model_id: string;
  model_family: string;
  architecture: string;
  version: string;
  weights_path: string;
  weights_exists: boolean;
  metrics: Record<string, any>;
  confusion_matrix: { labels?: string[]; matrix?: number[][] };
  is_active: boolean;
  notes: string;
  created_at: string;
}

export interface TrainingSession {
  job_id: string;
  model_family: string;
  architecture: string;
  dataset_id: number | null;
  status: string;
  current_epoch: number;
  total_epochs: number;
  batch_size: number;
  learning_rate: number;
  device_used: string;
  mixed_precision: boolean;
  best_val_metric: number;
  metrics_history: Array<Record<string, any>>;
  error_message: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface DatasetSummary {
  id: number;
  name: string;
  version: string;
  source_type: string;
  license_name: string;
  description: string;
  root_path: string;
  task_types: string[];
  item_count: number;
  splits: Record<string, number>;
  masks_available: number;
  class_distribution: Record<string, number>;
  integrity_verified: boolean;
  created_at: string;
}

export interface SystemStatus {
  version: string;
  offline_mode: boolean;
  setup_completed?: boolean;
  observations_count?: number;
  privacy: string;
  hardware: {
    cpu_percent: number;
    cpu_cores: number;
    ram_total_gb: number;
    ram_used_gb: number;
    ram_percent: number;
    gpu: {
      cuda_available: boolean;
      active_device: string;
      fallback_active: boolean;
      device_name: string;
      vram_total_gb: number;
      vram_allocated_mb: number;
      vram_reserved_mb: number;
      gpu_utilization_pct: number | null;
      recommended_batch_size: number;
    };
  };
  storage: {
    data_dir: string;
    database_size_mb: number;
    raw_images_mb: number;
    processed_images_mb: number;
    event_images_mb: number;
    timelapse_mb: number;
    datasets_mb: number;
    models_mb: number;
    total_app_storage_mb: number;
    total_app_storage_gb: number;
    max_storage_gb: number;
    disk_free_gb: number;
    disk_total_gb: number;
  };
  camera: CameraStatus;
  latest_inference_ms: number | null;
  training_active_jobs: number;
  active_models: Record<string, any>;
  online_weather?: any;
}
