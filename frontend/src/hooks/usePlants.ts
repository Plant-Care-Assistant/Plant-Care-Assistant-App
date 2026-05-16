"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { plantApi } from "@/lib/api";
import { CareType, UserPlant, UserPlantCreate, UserPlantUpdate } from "@/types";
import { savePlantImage } from "@/lib/utils/plantImages";
import { dataUrlToFile } from "@/lib/utils/dataUrl";

const PLANTS_KEY = ["plants"];
const CATALOG_KEY = ["plants", "catalog"];

export function useCatalogPlantQuery(catalogId: number | null | undefined) {
  return useQuery({
    queryKey: [...CATALOG_KEY, catalogId],
    queryFn: () => plantApi.getCatalogPlant(catalogId as number),
    enabled: catalogId != null,
    staleTime: 1000 * 60 * 60,
  });
}

export function usePlantsQuery(enabled = true) {
  return useQuery({
    queryKey: PLANTS_KEY,
    queryFn: () => plantApi.getPlants(),
    enabled,
  });
}

export function usePlantQuery(id?: number) {
  return useQuery({
    queryKey: [...PLANTS_KEY, id],
    queryFn: () => plantApi.getPlant(id as number),
    enabled: !!id,
  });
}

export function useIdentifyPlantMutation() {
  return useMutation({
    mutationFn: (file: File) => plantApi.identifyPlant(file),
  });
}

export function useAddPlantMutation() {
  const qc = useQueryClient();
  return useMutation<UserPlant, unknown, UserPlantCreate & { imageUrl?: string }>({
    mutationFn: ({ imageUrl: _, ...plant }) => plantApi.addPlant(plant),
    onSuccess: async (createdPlant, variables) => {
      if (variables.imageUrl && createdPlant.id) {
        // Optimistic local cache so the card has something to show before the gallery upload finishes.
        savePlantImage(createdPlant.id, variables.imageUrl);
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

export function useUpdatePlantMutation() {
  const qc = useQueryClient();
  return useMutation<UserPlant, unknown, { id: number; updates: UserPlantUpdate }>({
    mutationFn: ({ id, updates }) => plantApi.updatePlant(id, updates),
    onSuccess: (_, { id }) => {
      qc.invalidateQueries({ queryKey: [...PLANTS_KEY, id] });
      qc.invalidateQueries({ queryKey: PLANTS_KEY });
    },
    onError: (err) => console.error("updatePlant failed:", err),
  });
}

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

export function useRecordCareMutation(plantId: number) {
  const qc = useQueryClient();
  return useMutation<void, unknown, CareType>({
    mutationFn: (type: CareType) => plantApi.recordCare(plantId, type),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: [...CARE_HISTORY_KEY, plantId] });
    },
    onError: (err) => console.error("recordCare failed:", err),
  });
}
