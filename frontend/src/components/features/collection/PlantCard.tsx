'use client';

import { motion } from 'framer-motion';
import { Droplets, Sun } from 'lucide-react';

interface PlantCardProps {
  id: string;
  name: string;
  species?: string;
  imageUrl?: string;
  lightLevel?: 'low' | 'medium' | 'high';
  health: 'healthy' | 'needs-attention' | 'critical';
  /** Days remaining until the next watering; 0 = overdue/today. */
  daysUntilWater?: number | null;
  darkMode: boolean;
  onClick?: () => void;
}

const healthColors = {
  healthy: 'text-green-500',
  'needs-attention': 'text-yellow-500',
  critical: 'text-red-500',
};

const healthLabels = {
  healthy: 'Healthy',
  'needs-attention': 'Needs Care',
  critical: 'Critical',
};

function wateringLabel(daysUntilWater: number | null | undefined): string | null {
  if (daysUntilWater == null) return null;
  if (daysUntilWater <= 0) return 'Water now';
  if (daysUntilWater === 1) return 'Water tomorrow';
  return `Water in ${daysUntilWater}d`;
}

export const PlantCard: React.FC<PlantCardProps> = ({
  name,
  species,
  imageUrl,
  health,
  lightLevel,
  daysUntilWater,
  darkMode,
  onClick,
}) => {
  const overdue = daysUntilWater != null && daysUntilWater <= 0;
  const dueSoon = daysUntilWater != null && daysUntilWater > 0 && daysUntilWater <= 1;
  const wateringText = wateringLabel(daysUntilWater);

  // Overdue plants get a red ring to draw the eye in a grid of cards.
  const overdueRing = overdue ? 'ring-2 ring-red-500 ring-offset-2 ring-offset-transparent' : '';

  return (
    <motion.div
      onClick={onClick}
      className={`rounded-2xl overflow-hidden cursor-pointer transition-all ${
        darkMode
          ? 'bg-neutral-800 hover:bg-neutral-700/80'
          : 'bg-white hover:bg-neutral-50'
      } shadow-md hover:shadow-xl ${overdueRing}`}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      {/* Plant Image */}
      <div className={`w-full h-40 lg:h-48 relative ${
        darkMode ? 'bg-neutral-700' : 'bg-neutral-100'
      }`}>
        {imageUrl ? (
          <img
            src={imageUrl}
            alt={name}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <Droplets size={48} className={darkMode ? 'text-neutral-600' : 'text-neutral-300'} />
          </div>
        )}

        {/* Watering Urgency Badge (top-left) — only when overdue/due soon */}
        {wateringText && (overdue || dueSoon) && (
          <div
            className={`absolute top-2 left-2 px-2 py-1 rounded-full text-xs font-semibold backdrop-blur-sm flex items-center gap-1 ${
              overdue
                ? 'bg-red-500/90 text-white'
                : 'bg-amber-400/90 text-white'
            }`}
          >
            <Droplets size={12} fill="currentColor" />
            {wateringText}
          </div>
        )}

        {/* Health Badge (top-right) */}
        <div className={`absolute top-2 right-2 px-2 py-1 rounded-full text-xs font-semibold backdrop-blur-sm ${
          darkMode ? 'bg-neutral-900/70' : 'bg-white/70'
        } ${healthColors[health]}`}>
          {healthLabels[health]}
        </div>
      </div>

      {/* Plant Info */}
      <div className="p-4">
        <h3 className={`font-bold text-base lg:text-lg mb-1 truncate ${
          darkMode ? 'text-white' : 'text-gray-900'
        }`}>
          {name}
        </h3>
        {species && (
          <p className={`text-xs mb-3 opacity-70 truncate ${
            darkMode ? 'text-neutral-400' : 'text-neutral-600'
          }`}>
            {species}
          </p>
        )}

        {/* Quick Stats */}
        <div className="flex items-center gap-3 text-xs">
          {wateringText && (
            <div
              className={`flex items-center gap-1 ${
                overdue
                  ? 'text-red-500 font-semibold'
                  : dueSoon
                    ? 'text-amber-500 font-semibold'
                    : darkMode
                      ? 'text-neutral-400'
                      : 'text-neutral-600'
              }`}
            >
              <Droplets size={14} />
              <span>{wateringText}</span>
            </div>
          )}
          {lightLevel && (
            <div className={`flex items-center gap-1 ${
              darkMode ? 'text-neutral-400' : 'text-neutral-600'
            }`}>
              <Sun size={14} />
              <span className="capitalize">{lightLevel}</span>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
};
