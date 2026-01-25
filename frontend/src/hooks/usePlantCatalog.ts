import { plantApi } from "@/lib/api";
import {
  useQuery,
  useInfiniteQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query";

export function usePlantCatalog({
  pageSize = 20,
  search = "",
  filters = {},
} = {}) {
  // Infinite query for paginated plant catalog
  return useInfiniteQuery({
    queryKey: ["plant-catalog", search, filters, pageSize],
    queryFn: async ({ pageParam = 0 }) => {
      if (search) {
        const results = await plantApi.searchPlantCatalog(search);
        return results.slice(pageParam, pageParam + pageSize);
      }
      return plantApi.getPlantCatalog({ limit: pageSize, offset: pageParam });
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
    queryKey: ["plant-detail", plantId],
    queryFn: () => plantApi.getCatalogPlant(plantId),
    enabled: !!plantId,
  });
}

export function useAddToCollection(plantId?: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async () => {
      if (plantId == null) throw new Error("No plantId provided");
      await plantApi.addPlant({ catalogId: plantId });
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["plants"] }),
  });
}

export function usePlantImage(plantId: number) {
  return useQuery({
    queryKey: ["plant-image", plantId],
    queryFn: () => plantApi.getCatalogPlantImage(plantId),
    enabled: !!plantId,
  });
}
