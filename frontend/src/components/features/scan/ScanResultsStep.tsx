'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { ChevronLeft, Sparkles, AlertCircle, Sun, Droplet } from 'lucide-react';
import Image from 'next/image';
import type { ScanPlantData } from './ScanCameraModal';

interface ScanResultsStepProps {
  plantData: Partial<ScanPlantData>;
  onSubmit: (details: Partial<ScanPlantData>) => void;
  onBack: () => void;
  darkMode: boolean;
}

export function ScanResultsStep({
  plantData,
  onSubmit,
  onBack,
  darkMode,
}: ScanResultsStepProps) {
  const [formData, setFormData] = useState<Partial<ScanPlantData>>({
    name: plantData.name || '',
    species: plantData.species || '',
    location: plantData.location || '',
    lightLevel: plantData.lightLevel || 'medium',
    wateringFrequency: plantData.wateringFrequency || 3,
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  const lightLevels = [
    { value: 'low', label: 'Low Light', icon: '🌙' },
    { value: 'medium', label: 'Medium Light', icon: '☀️' },
    { value: 'high', label: 'Bright Light', icon: '🌞' },
  ] as const;

  const isAIIdentified = plantData.aiIdentified && (plantData.confidence || 0) > 50;

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* AI Result Banner */}
      {isAIIdentified ? (
        <div className={`rounded-xl p-4 flex items-start gap-3 ${
          darkMode ? 'bg-secondary/10 border border-secondary/20' : 'bg-secondary/10 border border-secondary/20'
        }`}>
          <Sparkles className="h-5 w-5 text-secondary flex-shrink-0 mt-0.5" />
          <div>
            <p className={`font-semibold text-sm ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
              AI Identified with {plantData.confidence}% confidence
            </p>
            <p className={`text-xs mt-1 ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
              We think this is a {plantData.name}. You can edit the details below if needed.
            </p>
          </div>
        </div>
      ) : (
        <div className={`rounded-xl p-4 flex items-start gap-3 ${
          darkMode ? 'bg-neutral-800 border border-neutral-700' : 'bg-neutral-100 border border-neutral-200'
        }`}>
          <AlertCircle className="h-5 w-5 text-amber-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className={`font-semibold text-sm ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
              Couldn't identify automatically
            </p>
            <p className={`text-xs mt-1 ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
              Please provide your plant's details manually.
            </p>
          </div>
        </div>
      )}

      {/* Plant Image Preview */}
      {plantData.imageUrl && (
        <div className="relative aspect-video w-full overflow-hidden rounded-2xl">
          <Image
            src={plantData.imageUrl}
            alt="Plant preview"
            fill
            className="object-cover"
          />
        </div>
      )}

      {/* Plant Name (Required) */}
      <div>
        <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
          Plant Name <span className="text-red-500">*</span>
        </label>
        <input
          type="text"
          required
          value={formData.name}
          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
          placeholder="e.g., My Monstera"
          className={`w-full rounded-xl border px-4 py-3 transition-colors ${
            darkMode
              ? 'bg-neutral-800 border-neutral-700 text-white placeholder:text-neutral-500 focus:border-secondary'
              : 'bg-white border-neutral-300 text-neutral-900 placeholder:text-neutral-400 focus:border-secondary'
          } focus:outline-none focus:ring-2 focus:ring-secondary/20`}
        />
      </div>

      {/* Species (Optional) */}
      <div>
        <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
          Species (optional)
        </label>
        <input
          type="text"
          value={formData.species}
          onChange={(e) => setFormData({ ...formData, species: e.target.value })}
          placeholder="e.g., Monstera Deliciosa"
          className={`w-full rounded-xl border px-4 py-3 transition-colors ${
            darkMode
              ? 'bg-neutral-800 border-neutral-700 text-white placeholder:text-neutral-500 focus:border-secondary'
              : 'bg-white border-neutral-300 text-neutral-900 placeholder:text-neutral-400 focus:border-secondary'
          } focus:outline-none focus:ring-2 focus:ring-secondary/20`}
        />
      </div>

      {/* Location (Optional) */}
      <div>
        <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
          Location (optional)
        </label>
        <input
          type="text"
          value={formData.location}
          onChange={(e) => setFormData({ ...formData, location: e.target.value })}
          placeholder="e.g., Living Room"
          className={`w-full rounded-xl border px-4 py-3 transition-colors ${
            darkMode
              ? 'bg-neutral-800 border-neutral-700 text-white placeholder:text-neutral-500 focus:border-secondary'
              : 'bg-white border-neutral-300 text-neutral-900 placeholder:text-neutral-400 focus:border-secondary'
          } focus:outline-none focus:ring-2 focus:ring-secondary/20`}
        />
      </div>

      {/* Light Level */}
      <div>
        <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
          Light Level
        </label>
        <div className="grid grid-cols-3 gap-2">
          {lightLevels.map((level) => (
            <button
              key={level.value}
              type="button"
              onClick={() => setFormData({ ...formData, lightLevel: level.value })}
              className={`rounded-xl px-3 py-3 text-sm font-medium transition-all ${
                formData.lightLevel === level.value
                  ? 'bg-secondary text-white'
                  : darkMode
                    ? 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
                    : 'bg-neutral-100 text-neutral-700 hover:bg-neutral-200'
              }`}
            >
              <div className="text-xl mb-1">{level.icon}</div>
              <div className="text-xs">{level.label.split(' ')[0]}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Watering Frequency */}
      <div>
        <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
          Water Every (days)
        </label>
        <input
          type="number"
          min="1"
          max="30"
          value={formData.wateringFrequency}
          onChange={(e) => setFormData({ ...formData, wateringFrequency: parseInt(e.target.value) || 3 })}
          className={`w-full rounded-xl border px-4 py-3 transition-colors ${
            darkMode
              ? 'bg-neutral-800 border-neutral-700 text-white focus:border-secondary'
              : 'bg-white border-neutral-300 text-neutral-900 focus:border-secondary'
          } focus:outline-none focus:ring-2 focus:ring-secondary/20`}
        />
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3 pt-4">
        <button
          type="button"
          onClick={onBack}
          className={`flex items-center gap-2 rounded-xl px-4 py-3 font-medium transition-colors ${
            darkMode
              ? 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
              : 'bg-neutral-200 text-neutral-700 hover:bg-neutral-300'
          }`}
        >
          <ChevronLeft className="h-4 w-4" />
          Back
        </button>

        <button
          type="submit"
          className="flex-1 rounded-xl bg-secondary px-4 py-3 font-medium text-white transition-colors hover:bg-secondary/90"
        >
          Continue to Summary
        </button>
      </div>
    </form>
  );
}
