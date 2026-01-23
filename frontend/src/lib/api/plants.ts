import { apiClient } from "./client";
import { Plant, PlantIdentification } from "@/types";

/**
 * Plant API endpoints
 */
export const plantApi = {
  /**
   * Get all plants in user's collection
   */
  async getPlants(): Promise<Plant[]> {
    const response = await apiClient.get<Plant[]>("/plants");
    return response.data;
  },

  /**
   * Get single plant by ID
   */
  async getPlant(id: string): Promise<Plant> {
    const response = await apiClient.get<Plant>(`/plants/${id}`);
    return response.data;
  },

  /**
   * Identify plant from image
   * @param file - Image file to analyze
   * @returns Plant identification result
   */
  async identifyPlant(file: File): Promise<PlantIdentification> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await apiClient.post<PlantIdentification>(
      "/plants/identify",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );
    return response.data;
  },

  /**
   * Add plant to collection
   */
  async addPlant(plant: Partial<Plant>): Promise<Plant> {
    const response = await apiClient.post<Plant>("/plants", plant);
    return response.data;
  },

  /**
   * Update plant
   */
  async updatePlant(id: string, updates: Partial<Plant>): Promise<Plant> {
    const response = await apiClient.patch<Plant>(`/plants/${id}`, updates);
    return response.data;
  },

  /**
   * Delete plant from collection
   */
  async deletePlant(id: string): Promise<void> {
    await apiClient.delete(`/plants/${id}`);
  },

  /**
   * Water plant (update last watered timestamp)
   */
  async waterPlant(id: string): Promise<Plant> {
    const response = await apiClient.post<Plant>(`/plants/${id}/water`);
    return response.data;
  },
};
