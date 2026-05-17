import { apiClient } from "./client";
import {
  AiCombinedResponse,
  AiHealthResponse,
  AiSpeciesPrediction,
  AiSpeciesResponse,
  CareHistory,
  CareType,
  CatalogPlant,
  PlantIdentification,
  UserPlant,
  UserPlantCreate,
  UserPlantImage,
  UserPlantUpdate,
} from "@/types";

export const plantApi = {
  async getPlants(): Promise<UserPlant[]> {
    const response = await apiClient.get<UserPlant[]>("/my-plants");
    return response.data;
  },

  async getPlant(id: number): Promise<UserPlant> {
    const response = await apiClient.get<UserPlant>(`/my-plants/${id}`);
    return response.data;
  },

  // Disease detection requires DISEASE_CHECKPOINT_PATH + YOLO_CHECKPOINT_PATH on the AI service.
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

  // Looks up a catalog entry by AI class_id (matches plants_catalog.plantsnet_id); null falls back to AI-only data.
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

  async getCatalogPlant(catalogId: number): Promise<CatalogPlant | null> {
    try {
      const response = await apiClient.get<CatalogPlant>(`/plants/${catalogId}`);
      return response.data;
    } catch {
      return null;
    }
  },

  // Routes to /ai/predict/combined when disease detection is wired, else /ai/predict; enriches with catalog data.
  async identifyPlant(file: File): Promise<PlantIdentification> {
    const { disease_detection_available } = await plantApi.getAiHealth();

    const formData = new FormData();
    formData.append("file", file);

    let top: AiSpeciesPrediction | null = null;
    let healthLabel: "healthy" | "diseased" | null = null;
    let healthConfidence: number | null = null;
    let diseases: Array<{ plant: string; condition: string; confidence: number }> | null = null;

    if (disease_detection_available) {
      const res = await fetch("/ai/predict/combined", { method: "POST", body: formData });
      if (!res.ok) throw new Error(`AI service error: ${res.status}`);
      const json: AiCombinedResponse = await res.json();
      // Species may be empty even when disease detection succeeded; surface whatever we got.
      top = json.species?.[0] ?? null;
      healthLabel = json.health?.label ?? null;
      healthConfidence = json.health?.confidence ?? null;
      diseases =
        json.diseases?.map((d) => {
          const [plant, condition] = d.disease.split("___");
          return {
            plant: (plant ?? d.disease).replace(/_/g, " "),
            condition: (condition ?? "").replace(/_/g, " "),
            confidence: d.confidence,
          };
        }) ?? null;

      // Discard a "diseased" verdict when the top disease probability is below 20%.
      // High health-classifier confidence alone is not enough — if the model cannot
      // confidently name any specific disease, the result is too unreliable to surface.
      const MIN_DISEASE_CONFIDENCE = 0.20;
      if (healthLabel === "diseased" && (diseases?.[0]?.confidence ?? 0) < MIN_DISEASE_CONFIDENCE) {
        healthLabel = null;
        healthConfidence = null;
        diseases = null;
      }

      if (!top && !healthLabel) {
        throw new Error("AI returned no species and no health verdict");
      }
    } else {
      const res = await fetch("/ai/predict", { method: "POST", body: formData });
      if (!res.ok) throw new Error(`AI service error: ${res.status}`);
      const json: AiSpeciesResponse = await res.json();
      if (!json.predictions?.length) throw new Error("No predictions returned");
      top = json.predictions[0];
    }

    const scientific = top
      ? (top.class_name ?? top.class_id).replace(/_/g, " ")
      : "";
    const catalog = top
      ? await plantApi.getCatalogPlantByPlantsnetId(top.class_id)
      : null;

    return {
      name: catalog?.common_name ?? (scientific ? scientific.split(" ")[0] : ""),
      scientificName: scientific,
      confidence: top ? Math.round(top.confidence * 100) : 0,
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

  async addPlant(plant: UserPlantCreate): Promise<UserPlant> {
    const response = await apiClient.post<UserPlant>("/my-plants", plant);
    return response.data;
  },

  async updatePlant(id: number, updates: UserPlantUpdate): Promise<UserPlant> {
    const response = await apiClient.patch<UserPlant>(`/my-plants/${id}`, updates);
    return response.data;
  },

  async deletePlant(id: number): Promise<void> {
    await apiClient.delete(`/my-plants/${id}`);
  },

  async listImages(plantId: number): Promise<UserPlantImage[]> {
    const res = await apiClient.get<UserPlantImage[]>(`/my-plants/${plantId}/images`);
    return res.data;
  },

  async uploadImage(plantId: number, file: File): Promise<UserPlantImage> {
    const formData = new FormData();
    formData.append("file", file);
    const res = await apiClient.post<UserPlantImage>(
      `/my-plants/${plantId}/images`,
      formData,
      { headers: { "Content-Type": "multipart/form-data" } },
    );
    return res.data;
  },

  async deleteImage(plantId: number, imageId: number): Promise<void> {
    await apiClient.delete(`/my-plants/${plantId}/images/${imageId}`);
  },

  async recordWatering(plantId: number): Promise<void> {
    await apiClient.post(`/my-plants/${plantId}/water`);
  },

  // For type='water' this is equivalent to the legacy /water endpoint.
  async recordCare(plantId: number, type: CareType): Promise<void> {
    await apiClient.post(`/my-plants/${plantId}/care`, { type });
  },

  async getCareHistory(plantId: number, days = 30): Promise<CareHistory> {
    const res = await apiClient.get<CareHistory>(
      `/my-plants/${plantId}/care-history?days=${days}`,
    );
    return res.data;
  },
};
