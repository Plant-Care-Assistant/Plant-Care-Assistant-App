import axios from "axios";
import { apiClient } from "./client";
import { Plant, PlantIdentification } from "@/types";

const aiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_AI_API_BASE_URL || "http://localhost:8001",
});

/**
 * Plant API endpoints
 */
export const plantApi = {
  /**
   * Get all plants in user's collection
   */
  async getPlants(): Promise<Plant[]> {
    const response = await apiClient.get<Plant[]>("/my-plants");
    return response.data;
  },

  /**
   * Get single plant by ID
   */
  async getPlant(id: string): Promise<Plant> {
    const response = await apiClient.get<Plant>(`/my-plants/${id}`);
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

    const response = await aiClient.post<{
      predictions: { class_id: string; class_name: string | null; confidence: number }[];
    }>("/predict", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });

    if (!response.data.predictions?.length) {
      throw new Error("No predictions returned");
    }
    const top = response.data.predictions[0];
    const scientific = (top.class_name ?? top.class_id).replace(/_/g, " ");
    const genus = scientific.split(" ")[0];
    return {
      name: genus,
      scientificName: scientific,
      confidence: Math.round(top.confidence * 100),
      careInstructions: [],
      temperature: "",
      light: "",
      wateringFrequency: 7,
    };
  },

  /**
   * Add plant to collection
   */
  async addPlant(plant: Partial<Plant>): Promise<Plant> {
    const response = await apiClient.post<Plant>("/my-plants", plant);
    return response.data;
  },

  /**
   * Update plant
   */
  async updatePlant(id: string, updates: Partial<Plant>): Promise<Plant> {
    const response = await apiClient.patch<Plant>(`/my-plants/${id}`, updates);
    return response.data;
  },

  /**
   * Delete plant from collection
   */
  async deletePlant(id: string): Promise<void> {
    await apiClient.delete(`/my-plants/${id}`);
  },
};
