'use client';

import { motion } from 'framer-motion';
import { ChevronLeft, Sparkles, MapPin, Sun, Droplet, Check } from 'lucide-react';
import Image from 'next/image';
import type { ScanPlantData } from './ScanCameraModal';

interface ScanConfirmStepProps {
  plantData: ScanPlantData;
  onConfirm: () => void;
  onViewOnly: () => void;
  onBack: () => void;
  darkMode: boolean;
}

export function ScanConfirmStep({
  plantData,
  onConfirm,
  onViewOnly,
  onBack,
  darkMode,
}: ScanConfirmStepProps) {
  const lightLevelLabels = {
    low: '🌙 Low Light',
    medium: '☀️ Medium Light',
    high: '🌞 Bright Light',
  };

  return (
    <div className="space-y-6 pb-30">
      {/* Title */}
      <div className="text-center">
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: 'spring', duration: 0.5 }}
          className="mx-auto mb-3 flex h-16 w-16 items-center justify-center rounded-full bg-secondary/10"
        >
          <Check className="h-8 w-8 text-secondary" />
        </motion.div>
        <h3 className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
          Ready to Add!
        </h3>
        <p className={`text-sm mt-1 ${darkMode ? 'text-neutral-400' : 'text-neutral-500'}`}>
          Review your plant details before adding to collection
        </p>
      </div>

      {/* Plant Card Preview */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={`overflow-hidden rounded-2xl border ${
          darkMode ? 'bg-neutral-800 border-neutral-700' : 'bg-white border-neutral-200'
        }`}
      >
        {/* Plant Image */}
        <div className="relative aspect-video w-full overflow-hidden">
          <Image
            src={plantData.imageUrl}
            alt={plantData.name}
            fill
            className="object-cover"
          />
          {plantData.aiIdentified && (
            <div className="absolute top-3 right-3 flex items-center gap-1.5 rounded-full bg-secondary px-3 py-1.5 text-xs font-medium text-white">
              <Sparkles className="h-3 w-3" />
              Identified
            </div>
          )}
        </div>

        {/* Plant Details */}
        <div className="p-4 space-y-3">
          {/* Name & Species */}
          <div>
            <h4 className={`text-lg font-bold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
              {plantData.name}
            </h4>
            {plantData.species && (
              <p className={`text-sm italic ${darkMode ? 'text-neutral-400' : 'text-neutral-500'}`}>
                {plantData.species}
              </p>
            )}
          </div>

          {/* Location */}
          {plantData.location && (
            <div className="flex items-center gap-2">
              <MapPin className={`h-4 w-4 ${darkMode ? 'text-neutral-400' : 'text-neutral-500'}`} />
              <span className={`text-sm ${darkMode ? 'text-neutral-300' : 'text-neutral-600'}`}>
                {plantData.location}
              </span>
            </div>
          )}

          {/* Care Info Grid */}
          <div className="grid grid-cols-2 gap-3 pt-2">
            {/* Light Level */}
            <div className={`rounded-xl p-3 ${darkMode ? 'bg-neutral-900' : 'bg-neutral-50'}`}>
              <div className="flex items-center gap-2 mb-1">
                <Sun className={`h-4 w-4 ${darkMode ? 'text-accent2' : 'text-primary'}`} />
                <span className={`text-xs font-medium ${darkMode ? 'text-neutral-400' : 'text-neutral-500'}`}>
                  Light
                </span>
              </div>
              <p className={`text-sm font-semibold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
                {lightLevelLabels[plantData.lightLevel]}
              </p>
            </div>

            {/* Watering */}
            <div className={`rounded-xl p-3 ${darkMode ? 'bg-neutral-900' : 'bg-neutral-50'}`}>
              <div className="flex items-center gap-2 mb-1">
                <Droplet className={`h-4 w-4 ${darkMode ? 'text-accent2' : 'text-primary'}`} />
                <span className={`text-xs font-medium ${darkMode ? 'text-neutral-400' : 'text-neutral-500'}`}>
                  Water
                </span>
              </div>
              <p className={`text-sm font-semibold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
                Every {plantData.wateringFrequency} days
              </p>
            </div>
          </div>

          {/* Confidence Score */}
          {plantData.aiIdentified && plantData.confidence && (
            <div className={`rounded-xl p-3 ${darkMode ? 'bg-neutral-900' : 'bg-neutral-50'}`}>
              <div className="flex items-center justify-between mb-2">
                <span className={`text-xs font-medium ${darkMode ? 'text-neutral-400' : 'text-neutral-500'}`}>
                  Confidence
                </span>
                <span className={`text-xs font-bold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
                  {plantData.confidence}%
                </span>
              </div>
              <div className={`h-2 rounded-full overflow-hidden ${darkMode ? 'bg-neutral-800' : 'bg-neutral-200'}`}>
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${plantData.confidence}%` }}
                  transition={{ duration: 1, ease: 'easeOut' }}
                  className="h-full bg-secondary"
                />
              </div>
            </div>
          )}
        </div>
      </motion.div>

      {/* Action Buttons - fixed bottom on mobile */}
      <div className="pt-3 pb-4 px-1 flex flex-col gap-3">
        <div className="flex gap-3">
          <button
            type="button"
            onClick={onBack}
            className={`flex items-center gap-2 rounded-xl px-6 py-3 font-medium transition-colors ${
              darkMode
                ? 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
                : 'bg-neutral-200 text-neutral-700 hover:bg-neutral-300'
            }`}
          >
            <ChevronLeft className="h-4 w-4" />
            Edit
          </button>

          <button
            type="button"
            onClick={onConfirm}
            className="flex-1 rounded-xl bg-secondary px-6 py-3 font-medium text-white transition-colors hover:bg-secondary/90"
          >
            Add to Collection
          </button>
        </div>

        <button
          type="button"
          onClick={onViewOnly}
          className={`w-full rounded-xl px-6 py-3 font-medium transition-colors ${
            darkMode
              ? 'text-neutral-400 hover:text-neutral-300 hover:bg-neutral-800/50'
              : 'text-neutral-600 hover:text-neutral-700 hover:bg-neutral-100'
          }`}
        >
          View Only (Don't Add)
        </button>
      </div>
    </div>
  );
}
