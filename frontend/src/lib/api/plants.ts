import { apiClient } from "./client";
import { PlantIdentification, UserPlant, UserPlantCreate, UserPlantUpdate } from "@/types";

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
   * Identify plant from image via nginx AI proxy
   * @param file - Image file to analyze
   * @returns Plant identification result
   */
  async identifyPlant(file: File): Promise<PlantIdentification> {
    const formData = new FormData();
    formData.append("file", file);

    // Use a relative URL so the request is routed through the active proxy:
    // Next.js rewrite in dev (/ai/ → localhost:8001) or nginx in production.
    const res = await fetch("/ai/predict", { method: "POST", body: formData });
    if (!res.ok) {
      throw new Error(`AI service error: ${res.status}`);
    }
    const json: { predictions: { class_id: string; class_name: string | null; confidence: number }[] } =
      await res.json();

    if (!json.predictions?.length) {
      throw new Error("No predictions returned");
    }
    const top = json.predictions[0];
    const scientific = (top.class_name ?? top.class_id).replace(/_/g, " ");
    const genus = scientific.split(" ")[0];
    return {
      name: genus,
      scientificName: scientific,
      confidence: Math.round(top.confidence * 100),
      // TODO: fetch care details from plant catalog once the endpoint is available
      careInstructions: [],
      temperature: "",
      light: "",
      wateringFrequency: 7,
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
