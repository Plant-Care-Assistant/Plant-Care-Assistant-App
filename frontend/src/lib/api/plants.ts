import { apiClient, aiClient } from "./client";
import type { Plant } from "@/lib/utils/plantFilters";
import type { PlantIdentification } from "@/types";

/**
 * Backend response types
 */
interface UserPlantResponse {
  id: number;
  plant_catalog_id: number | null;
  custom_name: string | null;
  note: string | null;
  created_at: string;
  age: string | null;
  fid: string | null;
}

interface PlantCatalogResponse {
  id: number;
  common_name: string;
  scientific_name: string | null;
  preferred_sunlight: string;
  preferred_temp_min: number | null;
  preferred_temp_max: number | null;
  air_humidity_req: string | null;
  soil_humidity_req: string | null;
  preferred_watering_interval_days: number | null;
}

interface AIPrediction {
  class_id: string;
  class_name: string | null;
  confidence: number;
}

interface AIIdentifyResponse {
  predictions: AIPrediction[];
  processing_time_ms: number;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL
  ? process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "").endsWith("/api")
    ? process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "")
    : `${process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "")}/api`
  : "/api";

function buildApiUrl(path: string) {
  const base = API_BASE_URL.endsWith("/")
    ? API_BASE_URL.slice(0, -1)
    : API_BASE_URL;
  return `${base}${path}`;
}

/**
 * Map backend light level to frontend format
 */
function mapLightLevel(
  sunlight: string,
): "low" | "medium" | "high" | undefined {
  const map: Record<string, "low" | "medium" | "high"> = {
    low: "low",
    medium: "medium",
    high: "high",
    full_sun: "high",
    partial_shade: "medium",
    shade: "low",
  };
  return map[sunlight.toLowerCase()];
}

/**
 * Convert backend UserPlant to frontend Plant format
 */
function adaptUserPlantToPlant(
  userPlant: UserPlantResponse,
  catalog?: PlantCatalogResponse,
): Plant {
  return {
    id: String(userPlant.id),
    catalogId: userPlant.plant_catalog_id ?? catalog?.id,
    name: userPlant.custom_name || catalog?.common_name || "Unknown Plant",
    species: catalog?.scientific_name || undefined,
    lastWatered: undefined,
    nextWatering: undefined,
    health: "healthy",
    lightLevel: catalog ? mapLightLevel(catalog.preferred_sunlight) : undefined,
    imageUrl: userPlant.fid
      ? buildApiUrl(`/my-plants/${userPlant.id}/image`)
      : userPlant.plant_catalog_id
        ? buildApiUrl(`/plants/${userPlant.plant_catalog_id}/image`)
        : undefined,
    wateringFrequency: catalog?.preferred_watering_interval_days || undefined,
  };
}

// Cache for plant catalog data
const catalogCache = new Map<number, PlantCatalogResponse>();

async function fetchCatalogPlant(
  catalogId: number,
): Promise<PlantCatalogResponse | undefined> {
  if (catalogCache.has(catalogId)) {
    return catalogCache.get(catalogId);
  }

  try {
    const response = await apiClient.get<PlantCatalogResponse>(
      `/plants/${catalogId}`,
    );
    catalogCache.set(catalogId, response.data);
    return response.data;
  } catch {
    return undefined;
  }
}

async function findPlantInCatalog(
  scientificName: string,
): Promise<PlantCatalogResponse | null> {
  try {
    const searchQuery = scientificName.replace(/_/g, " ");
    const response = await apiClient.get<PlantCatalogResponse[]>(
      "/plants/search",
      {
        params: { q: searchQuery },
      },
    );
    return response.data[0] ?? null;
  } catch {
    return null;
  }
}

async function uploadImageFromUrl(
  plantId: number,
  imageUrl: string,
): Promise<void> {
  if (
    !imageUrl.startsWith("data:") &&
    !imageUrl.startsWith("blob:") &&
    !imageUrl.startsWith("http")
  ) {
    return;
  }

  const response = await fetch(imageUrl);
  const blob = await response.blob();
  const file = new File([blob], "plant.jpg", {
    type: blob.type || "image/jpeg",
  });
  await plantApi.uploadPlantImage(String(plantId), file);
}

function parseCatalogId(value: unknown): number | undefined {
  if (
    typeof value === "number" &&
    Number.isFinite(value) &&
    value > 0 &&
    value < 1_000_000_000
  ) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed) && parsed > 0) {
      return parsed;
    }
  }
  return undefined;
}

/**
 * Plant API endpoints
 */
export const plantApi = {
  /**
   * Get all plants in user's collection
   */
  async getPlants(): Promise<Plant[]> {
    const response = await apiClient.get<UserPlantResponse[]>("/my-plants/");
    const userPlants = response.data;

    const plants: Plant[] = [];
    for (const userPlant of userPlants) {
      let catalog: PlantCatalogResponse | undefined;
      if (userPlant.plant_catalog_id != null) {
        catalog = await fetchCatalogPlant(userPlant.plant_catalog_id);
      }
      plants.push(adaptUserPlantToPlant(userPlant, catalog));
    }

    return plants;
  },

  /**
   * Get single plant by ID
   */
  async getPlant(id: string): Promise<Plant> {
    const response = await apiClient.get<UserPlantResponse>(`/my-plants/${id}`);
    const userPlant = response.data;

    let catalog: PlantCatalogResponse | undefined;
    if (userPlant.plant_catalog_id != null) {
      catalog = await fetchCatalogPlant(userPlant.plant_catalog_id);
    }

    return adaptUserPlantToPlant(userPlant, catalog);
  },

  /**
   * Add plant to collection
   */
  async addPlant(plant: Partial<Plant>): Promise<Plant> {
    const catalogId = parseCatalogId(plant.catalogId ?? plant.id);
    const payload = {
      custom_name: plant.name,
      plant_catalog_id: catalogId,
    };

    const response = await apiClient.post<UserPlantResponse>(
      "/my-plants/",
      payload,
    );
    const created = adaptUserPlantToPlant(response.data);

    if (plant.imageUrl) {
      try {
        await uploadImageFromUrl(response.data.id, plant.imageUrl);
      } catch {
        // Ignore image upload errors for now; plant is still created.
      }
    }

    return created;
  },

  /**
   * Update plant
   */
  async updatePlant(id: string, updates: Partial<Plant>): Promise<Plant> {
    const payload = {
      custom_name: updates.name,
      note: updates.location,
    };
    const response = await apiClient.patch<UserPlantResponse>(
      `/my-plants/${id}`,
      payload,
    );
    return adaptUserPlantToPlant(response.data);
  },

  /**
   * Delete plant from collection
   */
  async deletePlant(id: string): Promise<void> {
    await apiClient.delete(`/my-plants/${id}`);
  },

  /**
   * Upload plant image
   */
  async uploadPlantImage(id: string, file: File): Promise<void> {
    const formData = new FormData();
    formData.append("file", file);

    await apiClient.post(`/my-plants/${id}/image`, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
  },

  /**
   * Get plant catalog (all available plants)
   */
  async getPlantCatalog(params?: { limit?: number; offset?: number }) {
    const response = await apiClient.get<PlantCatalogResponse[]>("/plants", {
      params,
    });
    return response.data;
  },

  /**
   * Search plant catalog
   */
  async searchPlantCatalog(query: string) {
    const response = await apiClient.get<PlantCatalogResponse[]>(
      "/plants/search",
      {
        params: { q: query },
      },
    );
    return response.data;
  },

  /**
   * Get a single plant from catalog by ID
   */
  async getCatalogPlant(id: number) {
    const response = await apiClient.get<PlantCatalogResponse>(`/plants/${id}`);
    return response.data;
  },

  /**
   * Get a catalog plant image as Blob
   */
  async getCatalogPlantImage(id: number) {
    const response = await apiClient.get(`/plants/${id}/image`, {
      responseType: "blob",
    });
    return response.data as Blob;
  },

  /**
   * Identify plant from image via AI service
   */
  async identifyPlant(file: File): Promise<PlantIdentification> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await aiClient.post<AIIdentifyResponse>(
      "/predict",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      },
    );

    const topPrediction = response.data.predictions[0];
    const scientificName = topPrediction?.class_name ?? "Unknown";
    const catalogEntry = await findPlantInCatalog(scientificName);

    return {
      name: catalogEntry?.common_name ?? scientificName.replace(/_/g, " "),
      scientificName: scientificName.replace(/_/g, " "),
      confidence: topPrediction?.confidence ?? 0,
      careInstructions: [],
      temperature: "18-24C",
      light: catalogEntry?.preferred_sunlight ?? "Medium",
      wateringFrequency: catalogEntry?.preferred_watering_interval_days ?? 7,
      catalogId: catalogEntry?.id,
    };
  },
};
