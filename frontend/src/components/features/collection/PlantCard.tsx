'use client';

import { motion } from 'framer-motion';
import { Droplets, Sun, Calendar } from 'lucide-react';

interface PlantCardProps {
  id: string;
  name: string;
  species?: string;
  imageUrl?: string;
  lastWatered?: string;
  nextWatering?: string;
  lightLevel?: 'low' | 'medium' | 'high';
  health: 'healthy' | 'needs-attention' | 'critical';
  darkMode: boolean;
  onClick?: () => void;
}

export const PlantCard: React.FC<PlantCardProps> = ({
  name,
  species,
  imageUrl,
  lastWatered,
  health,
  lightLevel,
  darkMode,
  onClick
}) => {
  const healthColors = {
    healthy: 'text-green-500',
    'needs-attention': 'text-yellow-500',
    critical: 'text-red-500'
  };

  const healthLabels = {
    healthy: 'Healthy',
    'needs-attention': 'Needs Care',
    critical: 'Critical'
  };

  return (
    <motion.div
      onClick={onClick}
      className={`rounded-2xl overflow-hidden cursor-pointer transition-all ${
        darkMode 
          ? 'bg-neutral-800 hover:bg-neutral-700/80' 
          : 'bg-white hover:bg-neutral-50'
      } shadow-md hover:shadow-xl`}
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
        
        {/* Health Badge */}
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
          {lastWatered && (
            <div className={`flex items-center gap-1 ${
              darkMode ? 'text-neutral-400' : 'text-neutral-600'
            }`}>
              <Droplets size={14} />
              <span>{lastWatered}</span>
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
