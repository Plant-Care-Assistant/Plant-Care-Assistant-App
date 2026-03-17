"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { plantApi } from "@/lib/api";
import { UserPlant, UserPlantCreate } from "@/types";

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

/** Add a new plant then refetch list. */
export function useAddPlantMutation() {
  const qc = useQueryClient();
  return useMutation<UserPlant, unknown, UserPlantCreate>({
    mutationFn: (plant) => plantApi.addPlant(plant),
    onSuccess: () => qc.invalidateQueries({ queryKey: PLANTS_KEY }),
  });
}
