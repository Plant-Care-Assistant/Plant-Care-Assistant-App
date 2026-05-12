import { apiClient } from "./client";
import {
  AiCombinedResponse,
  AiHealthResponse,
  AiSpeciesPrediction,
  AiSpeciesResponse,
  CatalogPlant,
  PlantIdentification,
  UserPlant,
  UserPlantCreate,
  UserPlantUpdate,
} from "@/types";

/**
 * Plant API endpoints
 */
export const plantApi = {
  /**
   * Get all plants in user's collection
   */
  async getPlants(): Promise<UserPlant[]> {
    const response = await apiClient.get<UserPlant[]>("/my-plants");
    return response.data;
  },

  /**
   * Get single plant by ID
   */
  async getPlant(id: number): Promise<UserPlant> {
    const response = await apiClient.get<UserPlant>(`/my-plants/${id}`);
    return response.data;
  },

  /**
   * Check whether the AI service is up and whether disease detection is wired
   * (requires both DISEASE_CHECKPOINT_PATH and YOLO_CHECKPOINT_PATH on the AI service).
   */
  async getAiHealth(): Promise<AiHealthResponse> {
    try {
      const res = await fetch("/ai/health");
      if (!res.ok) throw new Error(`AI health ${res.status}`);
      return await res.json();
    } catch {
      return {
        status: "not_ready",
        device: "cpu",
        num_classes: 0,
        checkpoint_loaded: false,
        disease_detection_available: false,
      };
    }
  },

  /**
   * Look up a catalog entry by the AI service's class_id (matches plants_catalog.plantsnet_id).
   * Returns null on 404 (species not in catalog) or any error — caller falls back to AI-only data.
   */
  async getCatalogPlantByPlantsnetId(plantsnetId: string): Promise<CatalogPlant | null> {
    try {
      const response = await apiClient.get<CatalogPlant>(
        `/plants/by-plantsnet/${plantsnetId}`
      );
      return response.data;
    } catch {
      return null;
    }
  },

  /**
   * Identify plant from image. Routes through /ai/predict/combined when disease
   * detection is available, otherwise falls back to /ai/predict. Enriches the
   * result with catalog data (care fields, common name) when the AI's class_id
   * has a matching plants_catalog row.
   */
  async identifyPlant(file: File): Promise<PlantIdentification> {
    const { disease_detection_available } = await plantApi.getAiHealth();

    const formData = new FormData();
    formData.append("file", file);

    let top: AiSpeciesPrediction;
    let healthLabel: "healthy" | "diseased" | null = null;
    let healthConfidence: number | null = null;
    let diseases: Array<{ plant: string; condition: string; confidence: number }> | null = null;

    if (disease_detection_available) {
      const res = await fetch("/ai/predict/combined", { method: "POST", body: formData });
      if (!res.ok) throw new Error(`AI service error: ${res.status}`);
      const json: AiCombinedResponse = await res.json();
      if (!json.species?.length) throw new Error("No species predictions returned");
      top = json.species[0];
      healthLabel = json.health.label;
      healthConfidence = json.health.confidence;
      diseases = json.diseases.map((d) => {
        const [plant, condition] = d.disease.split("___");
        return {
          plant: (plant ?? d.disease).replace(/_/g, " "),
          condition: (condition ?? "").replace(/_/g, " "),
          confidence: d.confidence,
        };
      });
    } else {
      const res = await fetch("/ai/predict", { method: "POST", body: formData });
      if (!res.ok) throw new Error(`AI service error: ${res.status}`);
      const json: AiSpeciesResponse = await res.json();
      if (!json.predictions?.length) throw new Error("No predictions returned");
      top = json.predictions[0];
    }

    const scientific = (top.class_name ?? top.class_id).replace(/_/g, " ");
    const catalog = await plantApi.getCatalogPlantByPlantsnetId(top.class_id);

    return {
      name: catalog?.common_name ?? scientific.split(" ")[0],
      scientificName: scientific,
      confidence: Math.round(top.confidence * 100),
      careInstructions: [],
      temperature:
        catalog?.preferred_temp_min != null && catalog?.preferred_temp_max != null
          ? `${catalog.preferred_temp_min}–${catalog.preferred_temp_max}°C`
          : "",
      light: catalog?.preferred_sunlight ?? "medium",
      wateringFrequency: catalog?.preferred_watering_interval_days ?? 7,
      catalogId: catalog?.id ?? null,
      diseaseAvailable: disease_detection_available,
      healthLabel,
      healthConfidence,
      diseases,
    };
  },

  /**
   * Add plant to collection
   */
  async addPlant(plant: UserPlantCreate): Promise<UserPlant> {
    const response = await apiClient.post<UserPlant>("/my-plants", plant);
    return response.data;
  },

  /**
   * Update plant
   */
  async updatePlant(id: number, updates: UserPlantUpdate): Promise<UserPlant> {
    const response = await apiClient.patch<UserPlant>(`/my-plants/${id}`, updates);
    return response.data;
  },

  /**
   * Delete plant from collection
   */
  async deletePlant(id: number): Promise<void> {
    await apiClient.delete(`/my-plants/${id}`);
  },
};
