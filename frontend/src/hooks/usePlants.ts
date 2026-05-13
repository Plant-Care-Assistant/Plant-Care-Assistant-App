"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { plantApi } from "@/lib/api";
import { UserPlant, UserPlantCreate } from "@/types";
import { savePlantImage } from "@/lib/utils/plantImages";
import { dataUrlToFile } from "@/lib/utils/dataUrl";

const PLANTS_KEY = ["plants"];
const CATALOG_KEY = ["plants", "catalog"];

/** Fetch a catalog entry (care fields, common name) for a UserPlant.plant_catalog_id. */
export function useCatalogPlantQuery(catalogId: number | null | undefined) {
  return useQuery({
    queryKey: [...CATALOG_KEY, catalogId],
    queryFn: () => plantApi.getCatalogPlant(catalogId as number),
    enabled: catalogId != null,
    staleTime: 1000 * 60 * 60, // catalog rarely changes
  });
}

/** Fetch all plants in the user's collection. */
export function usePlantsQuery(enabled = true) {
  return useQuery({
    queryKey: PLANTS_KEY,
    queryFn: () => plantApi.getPlants(),
    enabled,
  });
}

/** Fetch a single plant by id. */
export function usePlantQuery(id?: number) {
  return useQuery({
    queryKey: [...PLANTS_KEY, id],
    queryFn: () => plantApi.getPlant(id as number),
    enabled: !!id,
  });
}

/** Identify a plant from an image file. */
export function useIdentifyPlantMutation() {
  return useMutation({
    mutationFn: (file: File) => plantApi.identifyPlant(file),
  });
}

/** Add a new plant with optional image, then refetch list. */
export function useAddPlantMutation() {
  const qc = useQueryClient();
  return useMutation<UserPlant, unknown, UserPlantCreate & { imageUrl?: string }>({
    mutationFn: ({ imageUrl: _, ...plant }) => plantApi.addPlant(plant),
    onSuccess: async (createdPlant, variables) => {
      if (variables.imageUrl && createdPlant.id) {
        // Optimistic local cache so the collection card has something to show
        // immediately, even before the gallery upload finishes.
        savePlantImage(createdPlant.id, variables.imageUrl);
        // Persist the captured scan to the plant gallery so it shows up on the
        // detail page (and survives device switches via SeaweedFS).
        try {
          const file = dataUrlToFile(variables.imageUrl, `scan-${Date.now()}.jpg`);
          await plantApi.uploadImage(createdPlant.id, file);
          qc.invalidateQueries({ queryKey: ["plants", "images", createdPlant.id] });
        } catch (err) {
          console.error("Failed to persist scan image to gallery:", err);
        }
      }
      qc.invalidateQueries({ queryKey: PLANTS_KEY });
    },
    onError: (err) => {
      console.error("addPlant failed:", err);
    },
  });
}

/** Delete a plant and invalidate the list + detail cache. */
export function useDeletePlantMutation() {
  const qc = useQueryClient();
  return useMutation<void, unknown, number>({
    mutationFn: (id: number) => plantApi.deletePlant(id),
    onSuccess: (_, id) => {
      qc.removeQueries({ queryKey: [...PLANTS_KEY, id] });
      qc.invalidateQueries({ queryKey: PLANTS_KEY });
    },
  });
}

// === Gallery ===

const IMAGES_KEY = ["plants", "images"] as const;

export function usePlantImagesQuery(plantId: number | null | undefined) {
  return useQuery({
    queryKey: [...IMAGES_KEY, plantId],
    queryFn: () => plantApi.listImages(plantId as number),
    enabled: plantId != null,
  });
}

export function useUploadPlantImageMutation(plantId: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => plantApi.uploadImage(plantId, file),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: [...IMAGES_KEY, plantId] });
    },
    onError: (err) => console.error("uploadImage failed:", err),
  });
}

export function useDeletePlantImageMutation(plantId: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (imageId: number) => plantApi.deleteImage(plantId, imageId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: [...IMAGES_KEY, plantId] });
    },
  });
}

// === Care history ===

const CARE_HISTORY_KEY = ["plants", "care-history"] as const;

export function useCareHistoryQuery(plantId: number | null | undefined) {
  return useQuery({
    queryKey: [...CARE_HISTORY_KEY, plantId],
    queryFn: () => plantApi.getCareHistory(plantId as number),
    enabled: plantId != null,
  });
}

export function useRecordWateringMutation(plantId: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => plantApi.recordWatering(plantId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: [...CARE_HISTORY_KEY, plantId] });
    },
    onError: (err) => console.error("recordWatering failed:", err),
  });
}
