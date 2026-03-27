'use client';

import { motion } from 'framer-motion';
import { useState, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import { CollectionHeader } from '@/components/features/collection/CollectionHeader';
import { CollectionSearch } from '@/components/features/collection/CollectionSearch';
import { CollectionFilters, FilterOption } from '@/components/features/collection/CollectionFilters';
import { PlantCard } from '@/components/features/collection/PlantCard';
import { ScanCameraModal, type ScanPlantData } from '@/components/features/scan';
import { Plus } from 'lucide-react';
import { useAddPlantMutation } from '@/hooks/usePlants';
import { getPlantImage } from '@/lib/utils/plantImages';
import { UserPlant } from '@/types';

export interface CollectionScreenProps {
  darkMode: boolean;
  plants: UserPlant[];
}

export function CollectionScreen({ darkMode, plants }: CollectionScreenProps) {
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState<FilterOption>('all');
  const [isAddPlantModalOpen, setIsAddPlantModalOpen] = useState(false);
  const addPlantMutation = useAddPlantMutation();

  const filteredPlants = useMemo(() => {
    let filtered = [...plants];

    // Search by custom_name or note
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(p =>
        p.custom_name?.toLowerCase().includes(query) ||
        p.note?.toLowerCase().includes(query)
      );
    }

    // Filters - with real data we don't have health status yet,
    // so "recent" sorts by created_at, others show all for now
    if (activeFilter === 'recent') {
      filtered = filtered.sort((a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );
    }

    return filtered;
  }, [plants, searchQuery, activeFilter]);

  const handlePlantClick = (plantId: number) => {
    router.push(`/plant/${plantId}`);
  };

  const handleAddPlant = () => {
    setIsAddPlantModalOpen(true);
  };

  const handleAddToCollection = (plant: ScanPlantData) => {
    addPlantMutation.mutate({
      custom_name: plant.name || 'Unknown Plant',
      note: plant.species || null,
      plant_catalog_id: null,
      imageUrl: plant.imageUrl,
    });
    setIsAddPlantModalOpen(false);
  };

  return (
    <>
      <ScanCameraModal
        isOpen={isAddPlantModalOpen}
        onClose={() => setIsAddPlantModalOpen(false)}
        onAddToCollection={handleAddToCollection}
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

          {/* Plant Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 lg:gap-6">
            {filteredPlants.map((plant, index) => (
              <motion.div
                key={plant.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <PlantCard
                  id={String(plant.id)}
                  name={plant.custom_name || 'Unnamed Plant'}
                  species={plant.note || undefined}
                  imageUrl={getPlantImage(plant.id)}
                  health="healthy"
                  darkMode={darkMode}
                  onClick={() => handlePlantClick(plant.id)}
                />
              </motion.div>
            ))}

            {/* Add Plant Card */}
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
          </div>

          {/* No Results State */}
          {filteredPlants.length === 0 && plants.length > 0 && (
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

        </div>
      </div>
    </>
  );
}
