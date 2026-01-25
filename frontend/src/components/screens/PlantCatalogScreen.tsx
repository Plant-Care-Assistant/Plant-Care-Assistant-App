import { useState, useMemo } from 'react';
import { usePlantCatalog, useAddToCollection } from '@/hooks/usePlantCatalog';
import { PlantCard } from '@/components/features/collection/PlantCard';
import { CollectionSearch } from '@/components/features/collection/CollectionSearch';
import { CollectionFilters, FilterOption } from '@/components/features/collection/CollectionFilters';
import { Plus } from 'lucide-react';
import { useRouter } from 'next/navigation';

export function PlantCatalogScreen({ darkMode }: { darkMode: boolean }) {
  const [search, setSearch] = useState('');
  const [activeFilter, setActiveFilter] = useState<FilterOption>('all');
  const pageSize = 20;
  const router = useRouter();
  const {
    data,
    isLoading,
    isError,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
  } = usePlantCatalog({ pageSize, search, filters: { filter: activeFilter } });

  // Remove global addToCollection, use per-plant below

  // Flatten paginated data
  const plants = useMemo(() =>
    data?.pages?.flat() || [],
    [data]
  );

  const handlePlantClick = (plantId: number) => {
    router.push(`/plant/${plantId}`);
  };

  return (
    <div className={`min-h-screen pb-24 lg:pb-8 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
      <div className="p-4 lg:p-6 max-w-7xl mx-auto">
        <h1 className="text-2xl font-bold mb-4">Plant Catalog</h1>
        <CollectionSearch value={search} onChange={setSearch} darkMode={darkMode} />
        <CollectionFilters activeFilter={activeFilter} onFilterChange={setActiveFilter} darkMode={darkMode} />

        {isLoading && <div className="py-12 text-center text-lg">Loading plants...</div>}
        {isError && <div className="py-12 text-center text-red-500">Failed to load catalog.</div>}

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 lg:gap-6">
          {plants.map((plant, idx) => {
            const addToCollection = useAddToCollection(plant.id);
            return (
              <div key={plant.id} className="relative">
                <PlantCard
                  {...plant}
                  health="healthy"
                  darkMode={darkMode}
                  onClick={() => handlePlantClick(plant.id)}
                />
                <button
                  className="absolute bottom-4 right-4 mt-2 px-3 py-1 rounded bg-secondary text-white text-xs font-semibold hover:bg-secondary/90 z-10"
                  onClick={e => {
                    e.stopPropagation();
                    addToCollection.mutate();
                  }}
                >
                  <Plus size={14} className="inline mr-1" /> Add to Collection
                </button>
              </div>
            );
          })}
        </div>

        {hasNextPage && (
          <div className="flex justify-center mt-8">
            <button
              className="px-6 py-2 rounded-xl font-semibold bg-secondary text-white hover:bg-secondary/90 disabled:opacity-50"
              onClick={() => fetchNextPage()}
              disabled={isFetchingNextPage}
            >
              {isFetchingNextPage ? 'Loading more...' : 'Load More'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
