export interface Token {
  access_token: string;
  token_type: string;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface RegisterData {
  username: string;
  email: string;
  password: string;
}

export interface User {
  id: string;
  username: string;
  email: string;
  name?: string;
  created_at: string;
}

export interface PlantIdentification {
  name: string;
  scientificName: string;
  confidence: number;
  careInstructions: string[];
  temperature: string;
  light: "low" | "medium" | "high";
  wateringFrequency: number;
  catalogId: number | null;
  diseaseAvailable: boolean;
  healthLabel: "healthy" | "diseased" | null;
  healthConfidence: number | null;
  diseases: Array<{ plant: string; condition: string; confidence: number }> | null;
}

export interface CatalogPlant {
  id: number;
  common_name: string;
  scientific_name: string | null;
  plantsnet_id: string | null;
  preferred_sunlight: "low" | "medium" | "high";
  preferred_temp_min: number | null;
  preferred_temp_max: number | null;
  air_humidity_req: "low" | "medium" | "high" | null;
  soil_humidity_req: "low" | "medium" | "high" | null;
  preferred_watering_interval_days: number | null;
}

export interface AiSpeciesPrediction {
  class_id: string;
  class_name: string | null;
  confidence: number;
}

export interface AiHealthSummary {
  label: "healthy" | "diseased";
  confidence: number;
  logit: number;
}

export interface AiDiseasePrediction {
  disease: string;
  confidence: number;
}

export interface AiLeafResult {
  leaf_index: number;
  bbox: [number, number, number, number] | null;
  health_logit: number;
  health_label: string;
  top_disease: string;
  top_disease_conf: number;
}

export interface AiSpeciesResponse {
  predictions: AiSpeciesPrediction[];
  processing_time_ms: number;
}

export interface AiCombinedResponse {
  species: AiSpeciesPrediction[];
  health: AiHealthSummary;
  diseases: AiDiseasePrediction[];
  leaf_count: number;
  used_full_image_fallback: boolean;
  leaf_results: AiLeafResult[];
  processing_time_ms: number;
}

export interface AiHealthResponse {
  status: "healthy" | "not_ready";
  device: string;
  num_classes: number;
  checkpoint_loaded: boolean;
  disease_detection_available: boolean;
}

export interface UserPlantDisease {
  plant: string;
  condition: string;
  confidence: number;
}

export interface UserPlant {
  id: number;
  plant_catalog_id: number | null;
  custom_name: string | null;
  note: string | null;
  created_at: string;
  sprouted_at: string | null;

  scientific_name: string | null;
  preferred_sunlight: "low" | "medium" | "high" | null;
  preferred_temp_min: number | null;
  preferred_temp_max: number | null;
  air_humidity_req: "low" | "medium" | "high" | null;
  soil_humidity_req: "low" | "medium" | "high" | null;
  preferred_watering_interval_days: number | null;

  last_health_label: "healthy" | "diseased" | null;
  last_health_confidence: number | null;
  last_health_check_at: string | null;
  last_diseases: UserPlantDisease[] | null;

  // Days until next watering: 0 = due today/overdue.
  last_watered_at: string | null;
  days_until_water: number | null;
}

export interface UserPlantCreate {
  plant_catalog_id?: number | null;
  custom_name?: string | null;
  note?: string | null;
  sprouted_at?: string | null;

  last_health_label?: "healthy" | "diseased" | null;
  last_health_confidence?: number | null;
  last_health_check_at?: string | null;
  last_diseases?: UserPlantDisease[] | null;
}

export interface UserPlantUpdate {
  plant_catalog_id?: number | null;
  custom_name?: string | null;
  note?: string | null;
  sprouted_at?: string | null;

  scientific_name?: string | null;
  preferred_sunlight?: "low" | "medium" | "high" | null;
  preferred_temp_min?: number | null;
  preferred_temp_max?: number | null;
  air_humidity_req?: "low" | "medium" | "high" | null;
  soil_humidity_req?: "low" | "medium" | "high" | null;
  preferred_watering_interval_days?: number | null;

  last_health_label?: "healthy" | "diseased" | null;
  last_health_confidence?: number | null;
  last_health_check_at?: string | null;
  last_diseases?: UserPlantDisease[] | null;
}

export interface UserPlantImage {
  id: number;
  user_plant_id: number;
  fid: string;
  uploaded_at: string;
}

// Watering is the historical default; the rest let weekly-watering users still build streaks.
export type CareType =
  | "water"
  | "mist"
  | "fertilize"
  | "prune"
  | "rotate"
  | "inspect"
  | "other";

export interface CareEvent {
  timestamp: string;
  type: CareType;
}

export interface DailyCare {
  // ISO YYYY-MM-DD.
  date: string;
  types: CareType[];
}

export interface CareHistory {
  waterings: string[];
  events: CareEvent[];
  current_streak_days: number;
  unique_days_last_week: number;
  // Fixed 7-day strip ending today, oldest first.
  daily_last_week: DailyCare[];
}

export interface ApiError {
  message: string;
  detail?: string;
  status?: number;
}
