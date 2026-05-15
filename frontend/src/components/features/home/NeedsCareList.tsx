'use client';

import { motion } from 'framer-motion';
import { Droplet, AlertCircle, CheckCircle2 } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { UserPlant } from '@/types';
import { getPlantImage } from '@/lib/utils/plantImages';

interface NeedsCareListProps {
  /** Plants where days_until_water <= 0 (overdue or due today). */
  plants: UserPlant[];
  darkMode: boolean;
}

/** Compact tappable row used in the home "Needs care today" list. */
function PlantRow({ plant, darkMode, onClick }: { plant: UserPlant; darkMode: boolean; onClick: () => void }) {
  const imageUrl = getPlantImage(plant.id);
  const diseased = plant.last_health_label === 'diseased';

  return (
    <motion.button
      onClick={onClick}
      whileHover={{ scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
      className={`w-full flex items-center gap-3 p-3 rounded-2xl text-left transition-colors ${
        darkMode ? 'bg-neutral-800 hover:bg-neutral-700/70' : 'bg-white hover:bg-neutral-50'
      } shadow-sm`}
    >
      <div className={`w-12 h-12 rounded-xl overflow-hidden flex-shrink-0 ${
        darkMode ? 'bg-neutral-700' : 'bg-neutral-100'
      }`}>
        {imageUrl ? (
          <img src={imageUrl} alt={plant.custom_name ?? 'plant'} className="w-full h-full object-cover" />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <Droplet size={20} className={darkMode ? 'text-neutral-500' : 'text-neutral-400'} />
          </div>
        )}
      </div>

      <div className="flex-1 min-w-0">
        <p className={`font-semibold truncate ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
          {plant.custom_name || 'Unnamed plant'}
        </p>
        <p className="text-xs text-red-500 font-medium">Needs water today</p>
      </div>

      {diseased && (
        <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
      )}
    </motion.button>
  );
}

export function NeedsCareList({ plants, darkMode }: NeedsCareListProps) {
  const router = useRouter();

  if (plants.length === 0) {
    return (
      <div
        className={`flex items-center gap-3 p-4 rounded-2xl ${
          darkMode ? 'bg-neutral-800' : 'bg-green-50'
        }`}
      >
        <CheckCircle2 className="h-5 w-5 text-green-500 flex-shrink-0" />
        <p className={`text-sm ${darkMode ? 'text-neutral-300' : 'text-green-800'}`}>
          All caught up — no plants need watering today.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      {plants.map((p) => (
        <PlantRow
          key={p.id}
          plant={p}
          darkMode={darkMode}
          onClick={() => router.push(`/plant/${p.id}`)}
        />
      ))}
    </div>
  );
}
