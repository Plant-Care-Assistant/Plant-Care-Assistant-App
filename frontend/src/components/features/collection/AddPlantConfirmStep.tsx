'use client';

import { motion } from 'framer-motion';
import { ArrowLeft, Check } from 'lucide-react';
import { PlantFormData } from './AddPlantModal';

interface AddPlantConfirmStepProps {
  plantData: PlantFormData;
  onConfirm: () => void;
  onBack: () => void;
  darkMode: boolean;
}

export function AddPlantConfirmStep({
  plantData,
  onConfirm,
  onBack,
  darkMode,
}: AddPlantConfirmStepProps) {
  const getLightLabel = (level: string) => {
    switch (level) {
      case 'low':
        return '🌑 Low Light';
      case 'high':
        return '🌞 Bright Light';
      default:
        return '☀️ Medium Light';
    }
  };

  const infoItem = (label: string, value: string) => (
    <div className={`flex justify-between rounded-lg p-3 ${darkMode ? 'bg-neutral-800' : 'bg-neutral-100'}`}>
      <span className={darkMode ? 'text-neutral-400' : 'text-neutral-600'}>{label}</span>
      <span className={`font-medium ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
        {value}
      </span>
    </div>
  );

  return (
    <div className="space-y-4">
      {/* Plant Image Preview */}
      {plantData.imageUrl && (
        <div className="relative h-48 w-full overflow-hidden rounded-2xl bg-neutral-200 dark:bg-neutral-800">
          <img
            src={plantData.imageUrl}
            alt={plantData.name}
            className="h-full w-full object-cover"
          />
        </div>
      )}

      {/* Plant Info */}
      <div className="space-y-2">
        <h3 className={`text-xl font-semibold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
          {plantData.name}
        </h3>
        {plantData.species && (
          <p className={darkMode ? 'text-neutral-400' : 'text-neutral-600'}>
            {plantData.species}
          </p>
        )}
      </div>

      {/* Details Summary */}
      <div className="space-y-2">
        {plantData.location && infoItem('📍 Location', plantData.location)}
        {infoItem('💡 Light Level', getLightLabel(plantData.lightLevel))}
        {infoItem('💧 Water Every', `${plantData.wateringFrequency} days`)}
      </div>

      {/* Confirmation Message */}
      <div
        className={`rounded-xl border px-4 py-3 text-sm ${
          darkMode
            ? 'border-green-500/50 bg-green-500/10 text-green-400'
            : 'border-green-500/30 bg-green-50 text-green-700'
        }`}
      >
        <p>✨ Your plant is ready to be added to your collection!</p>
      </div>

      {/* Buttons */}
      <div className="flex gap-3 pt-4">
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={onBack}
          className={`flex flex-1 items-center justify-center gap-2 rounded-lg px-4 py-2 transition-colors ${
            darkMode
              ? 'border border-neutral-700 text-neutral-300 hover:bg-neutral-800'
              : 'border border-neutral-300 text-neutral-700 hover:bg-neutral-100'
          }`}
        >
          <ArrowLeft className="h-4 w-4" />
          Back
        </motion.button>

        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={onConfirm}
          className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-secondary px-4 py-2 font-medium text-white transition-colors hover:opacity-90"
        >
          <Check className="h-4 w-4" />
          Add to Collection
        </motion.button>
      </div>
    </div>
  );
}
