import { useState } from 'react';
import { PlantsAPI } from '@/lib/apiClient';
import { useQuery, useInfiniteQuery, useMutation, useQueryClient } from '@tanstack/react-query';

export function usePlantCatalog({ pageSize = 20, search = '', filters = {} } = {}) {
  // Infinite query for paginated plant catalog
  return useInfiniteQuery({
    queryKey: ['plant-catalog', search, filters, pageSize],
    queryFn: async ({ pageParam = 0 }) => {
      const params: Record<string, any> = { limit: pageSize, offset: pageParam, ...filters };
      if (search) params.q = search;
      const data = await PlantsAPI.list(params);
      return data;
    },
    getNextPageParam: (lastPage, allPages) => {
      if (!lastPage || lastPage.length < pageSize) return undefined;
      return allPages.flat().length;
    },
    initialPageParam: 0,
  });
}

export function usePlantDetail(plantId: number) {
  return useQuery({
    queryKey: ['plant-detail', plantId],
    queryFn: () => PlantsAPI.get(plantId),
    enabled: !!plantId,
  });
}

export function useAddToCollection(plantId?: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async () => {
      if (plantId == null) throw new Error('No plantId provided');
      await PlantsAPI.addToCollection(plantId);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['plants'] }),
  });
}

export function usePlantImage(plantId: number) {
  return useQuery({
    queryKey: ['plant-image', plantId],
    queryFn: () => PlantsAPI.getImage(plantId),
    enabled: !!plantId,
  });
}
