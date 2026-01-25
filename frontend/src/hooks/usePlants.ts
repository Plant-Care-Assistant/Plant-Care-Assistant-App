"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { plantApi } from "@/lib/api";
import type { Plant } from "@/lib/utils/plantFilters";
import type { PlantIdentification } from "@/types";

const PLANTS_KEY = ["plants"];

/** Fetch all plants in the user's collection. */
export function usePlantsQuery(enabled = true) {
  return useQuery<Plant[], Error>({
    queryKey: PLANTS_KEY,
    queryFn: () => plantApi.getPlants(),
    enabled,
  });
}

/** Fetch a single plant by id. */
export function usePlantQuery(id?: string) {
  return useQuery<Plant, Error>({
    queryKey: [...PLANTS_KEY, id],
    queryFn: () => plantApi.getPlant(id as string),
    enabled: !!id,
  });
}

/** Identify a plant from an image file. */
export function useIdentifyPlantMutation() {
  return useMutation<PlantIdentification, unknown, File>({
    mutationFn: (file) => plantApi.identifyPlant(file),
  });
}

/** Add a new plant then refetch list. */
export function useAddPlantMutation() {
  const qc = useQueryClient();
  return useMutation<Plant, unknown, Partial<Plant>>({
    mutationFn: (plant) => plantApi.addPlant(plant),
    onSuccess: () => qc.invalidateQueries({ queryKey: PLANTS_KEY }),
  });
}

/** Update plant details. */
export function useUpdatePlantMutation() {
  const qc = useQueryClient();
  return useMutation<Plant, unknown, { id: string; updates: Partial<Plant> }>({
    mutationFn: ({ id, updates }) => plantApi.updatePlant(id, updates),
    onSuccess: () => qc.invalidateQueries({ queryKey: PLANTS_KEY }),
  });
}

/** Delete plant from collection. */
export function useDeletePlantMutation() {
  const qc = useQueryClient();
  return useMutation<void, unknown, string>({
    mutationFn: (id) => plantApi.deletePlant(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: PLANTS_KEY }),
  });
}

/** Upload plant image. */
export function useUploadPlantImageMutation() {
  const qc = useQueryClient();
  return useMutation<void, unknown, { id: string; file: File }>({
    mutationFn: ({ id, file }) => plantApi.uploadPlantImage(id, file),
    onSuccess: () => qc.invalidateQueries({ queryKey: PLANTS_KEY }),
  });
}
