"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { plantApi } from "@/lib/api";
import { Plant, PlantIdentification } from "@/types";

const PLANTS_KEY = ["plants"];

/** Fetch all plants in the user's collection. */
export function usePlantsQuery(enabled = true) {
  return useQuery({
    queryKey: PLANTS_KEY,
    queryFn: () => plantApi.getPlants(),
    enabled,
  });
}

/** Fetch a single plant by id. */
export function usePlantQuery(id?: string) {
  return useQuery({
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

/** Water a plant and refresh cache. */
export function useWaterPlantMutation() {
  const qc = useQueryClient();
  return useMutation<Plant, unknown, string>({
    mutationFn: (id) => plantApi.waterPlant(id),
    onSuccess: (updated) => {
      qc.setQueryData<Plant[]>(PLANTS_KEY, (prev) =>
        prev?.map((p) => (p.id === updated.id ? updated : p)) || prev,
      );
    },
  });
}
