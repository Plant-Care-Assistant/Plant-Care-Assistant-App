/**
 * Core domain types for the Plant Care application
 */

/**
 * Authentication token response
 */
export interface Token {
  access_token: string;
  token_type: string;
}

/**
 * User authentication credentials
 */
export interface LoginCredentials {
  username: string;
  password: string;
}

/**
 * User registration data
 */
export interface RegisterData {
  username: string;
  email: string;
  password: string;
}

/**
 * Authenticated user information
 */
export interface User {
  id: string;
  username: string;
  email: string;
  name?: string;
  created_at: string;
}

/**
 * Plant identification result from AI
 */
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

/**
 * Plant catalog entry returned from GET /plants/by-plantsnet/{id} or /plants/{id}.
 */
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

/**
 * AI service response shapes (port :8001, proxied via /ai)
 */
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

/**
 * Disease snapshot stored on user_plants. Shape mirrors what AI service
 * returns in /predict/combined response after mapping.
 */
export interface UserPlantDisease {
  plant: string;
  condition: string;
  confidence: number;
}

/**
 * User plant as returned by the backend (GET /my-plants)
 */
export interface UserPlant {
  id: number;
  plant_catalog_id: number | null;
  custom_name: string | null;
  note: string | null;
  created_at: string;
  sprouted_at: string | null;

  last_health_label: "healthy" | "diseased" | null;
  last_health_confidence: number | null;
  last_health_check_at: string | null;
  last_diseases: UserPlantDisease[] | null;
}

/**
 * Payload for creating a new user plant (POST /my-plants)
 */
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

/**
 * Payload for updating a user plant (PATCH /my-plants/:id)
 */
export interface UserPlantUpdate {
  plant_catalog_id?: number | null;
  custom_name?: string | null;
  note?: string | null;
  sprouted_at?: string | null;

  last_health_label?: "healthy" | "diseased" | null;
  last_health_confidence?: number | null;
  last_health_check_at?: string | null;
  last_diseases?: UserPlantDisease[] | null;
}

/**
 * Single photo of a user plant from the gallery (multi-image per plant).
 */
export interface UserPlantImage {
  id: number;
  user_plant_id: number;
  fid: string;
  uploaded_at: string;
}

/**
 * Watering history snapshot for plant detail widgets.
 */
export interface CareHistory {
  waterings: string[];
  current_streak_days: number;
  unique_days_last_week: number;
}

/**
 * API error response
 */
export interface ApiError {
  message: string;
  detail?: string;
  status?: number;
}
