"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { plantApi } from "@/lib/api";
import { UserPlant, UserPlantCreate } from "@/types";
import { savePlantImage } from "@/lib/utils/plantImages";

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
    onSuccess: (createdPlant, variables) => {
      if (variables.imageUrl && createdPlant.id) {
        savePlantImage(createdPlant.id, variables.imageUrl);
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
