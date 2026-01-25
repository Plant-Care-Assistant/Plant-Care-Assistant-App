'use client';

import { motion } from 'framer-motion';
import { useState, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import { CollectionHeader } from '@/components/features/collection/CollectionHeader';
import { CollectionSearch } from '@/components/features/collection/CollectionSearch';
import { CollectionFilters, FilterOption } from '@/components/features/collection/CollectionFilters';
import { PlantCard } from '@/components/features/collection/PlantCard';
import { AddPlantModal, type PlantFormData } from '@/components/features/collection/AddPlantModal';
import { Plus } from 'lucide-react';
import { filterAndSearchPlants, type Plant } from '@/lib/utils/plantFilters';
import { usePlantsQuery, useAddPlantMutation, useWaterPlantMutation } from '@/hooks/usePlants';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { UserPlantsAPI, UserPlantUpdate } from '@/lib/apiClient';

export interface CollectionScreenProps {
  darkMode: boolean;
}

export function CollectionScreen({ darkMode }: CollectionScreenProps) {
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState<FilterOption>('all');
  const [isAddPlantModalOpen, setIsAddPlantModalOpen] = useState(false);
  const queryClient = useQueryClient();

  // Fetch user plants from backend
  const { data: plants = [], isLoading, isError } = usePlantsQuery();

  // Add plant mutation (optimistic)
  const addPlantMutation = useAddPlantMutation();

  // Edit plant mutation
  const editPlantMutation = useMutation({
    mutationFn: ({ id, updates }: { id: number; updates: Partial<UserPlantUpdate> }) => UserPlantsAPI.update(id, updates),
    onMutate: async ({ id, updates }) => {
      await queryClient.cancelQueries({ queryKey: ['plants'] });
      const previous = queryClient.getQueryData<Plant[]>(['plants']);
      queryClient.setQueryData<Plant[]>(['plants'], (old) =>
        old?.map((p) => (p.id === id ? { ...p, ...updates } : p)) || old
      );
      return { previous };
    },
    onError: (err, variables, context) => {
      if (context?.previous) queryClient.setQueryData(['plants'], context.previous);
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: ['plants'] }),
  });

  // Delete plant mutation
  const deletePlantMutation = useMutation({
    mutationFn: (id: number) => UserPlantsAPI.delete(id),
    onMutate: async (id) => {
      await queryClient.cancelQueries({ queryKey: ['plants'] });
      const previous = queryClient.getQueryData<Plant[]>(['plants']);
      queryClient.setQueryData<Plant[]>(['plants'], (old) => old?.filter((p) => p.id !== id) || old);
      return { previous };
    },
    onError: (err, variables, context) => {
      if (context?.previous) queryClient.setQueryData(['plants'], context.previous);
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: ['plants'] }),
  });

  // Image upload mutation
  const uploadImageMutation = useMutation({
    mutationFn: ({ plantId, file }: { plantId: number; file: File }) => UserPlantsAPI.uploadImage(plantId, file),
    onSettled: () => queryClient.invalidateQueries({ queryKey: ['plants'] }),
  });

  // Water/care status mutation
  const waterPlantMutation = useWaterPlantMutation();

  // Map API plants to display plants with correct health type
  const displayPlants = useMemo(() =>
    plants.map((p) => ({
      ...p,
      health:
        typeof p.health === 'number'
          ? (p.health >= 80
              ? 'healthy'
              : p.health >= 50
              ? 'needs-attention'
              : 'critical') as 'healthy' | 'needs-attention' | 'critical'
          : (p.health as 'healthy' | 'needs-attention' | 'critical'),
    })),
    [plants]
  );

  // Search/filter logic (client-side for now)
  const filteredPlants = useMemo(() => {
    return filterAndSearchPlants(displayPlants, activeFilter, searchQuery);
  }, [displayPlants, searchQuery, activeFilter]);

  const handlePlantClick = (plantId: number) => {
    router.push(`/plant/${plantId}`);
  };

  const handleAddPlant = () => {
    setIsAddPlantModalOpen(true);
  };

  const handleSavePlant = async (plantData: PlantFormData) => {
    // Add plant to backend (optimistic)
    await addPlantMutation.mutateAsync({ ...plantData });
    setIsAddPlantModalOpen(false);
  };

  // TODO: Wire up edit, delete, image upload, and care status to UI controls

  return (
    <>
      <AddPlantModal
        isOpen={isAddPlantModalOpen}
        onClose={() => setIsAddPlantModalOpen(false)}
        onSave={handleSavePlant}
        darkMode={darkMode}
      />

      <div className={`min-h-screen pb-24 lg:pb-8 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
        <div className="p-4 lg:p-6 max-w-7xl mx-auto">
          {/* Header */}
          <CollectionHeader plantCount={plants.length} darkMode={darkMode} />

          {/* Search */}
          <CollectionSearch 
            value={searchQuery}
            onChange={setSearchQuery}
            darkMode={darkMode}
          />

          {/* Filters */}
          <CollectionFilters
            activeFilter={activeFilter}
            onFilterChange={setActiveFilter}
            darkMode={darkMode}
          />

          {/* Loading/Error States */}
          {isLoading && (
            <div className="py-12 text-center text-lg">Loading your plants...</div>
          )}
          {isError && (
            <div className="py-12 text-center text-red-500">Failed to load plants. Please try again.</div>
          )}

          {/* Plant Grid */}
          {!isLoading && !isError && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 lg:gap-6">
              {filteredPlants.map((plant, index) => (
                <motion.div
                  key={plant.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <PlantCard
                    {...plant}
                    darkMode={darkMode}
                    onClick={() => handlePlantClick(plant.id)}
                  />
                </motion.div>
              ))}

              {/* Add Plant Card - only show if there are plants */}
              {filteredPlants.length > 0 && (
                <motion.button
                  onClick={handleAddPlant}
                  className={`rounded-2xl border-2 border-dashed min-h-[280px] lg:min-h-[320px] flex flex-col items-center justify-center gap-3 transition-all ${
                    darkMode
                      ? 'border-neutral-700 hover:border-secondary hover:bg-neutral-800/30'
                      : 'border-neutral-300 hover:border-secondary hover:bg-neutral-50'
                  }`}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
                    darkMode ? 'bg-neutral-800' : 'bg-neutral-100'
                  }`}>
                    <Plus size={32} className="text-secondary" />
                  </div>
                  <p className={`font-semibold ${
                    darkMode ? 'text-neutral-300' : 'text-neutral-700'
                  }`}>
                    Add New Plant
                  </p>
                </motion.button>
              )}
            </div>
          )}

          {/* No Results State */}
          {!isLoading && !isError && filteredPlants.length === 0 && plants.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center py-12"
            >
              <p className={`text-lg ${
                darkMode ? 'text-neutral-400' : 'text-neutral-600'
              }`}>
                No plants found matching your search
              </p>
            </motion.div>
          )}

          {/* Empty State (if no plants) */}
          {!isLoading && !isError && plants.length === 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center py-12"
            >
              <div className={`w-24 h-24 rounded-full mx-auto mb-4 flex items-center justify-center ${
                darkMode ? 'bg-neutral-800' : 'bg-neutral-100'
              }`}>
                <Plus size={48} className="text-secondary" />
              </div>
              <h3 className={`text-xl font-bold mb-2 ${
                darkMode ? 'text-white' : 'text-gray-900'
              }`}>
                Start Your Collection
              </h3>
              <p className={`mb-6 ${
                darkMode ? 'text-neutral-400' : 'text-neutral-600'
              }`}>
                Add your first plant to begin tracking their care
              </p>
              <button
                onClick={handleAddPlant}
                className="px-6 py-3 rounded-xl font-semibold transition-colors bg-secondary text-white hover:bg-secondary/90"
              >
                Add Your First Plant
              </button>
            </motion.div>
          )}
        </div>
      </div>
    </>
  );
}
