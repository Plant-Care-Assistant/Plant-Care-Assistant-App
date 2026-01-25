'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft } from 'lucide-react';
import { PlantFormData } from './AddPlantModal';

interface AddPlantDetailsStepProps {
  plantData: Partial<PlantFormData>;
  onSubmit: (details: Partial<PlantFormData>) => void;
  onBack: () => void;
  darkMode: boolean;
}

export function AddPlantDetailsStep({
  plantData,
  onSubmit,
  onBack,
  darkMode,
}: AddPlantDetailsStepProps) {
  const [formData, setFormData] = useState<Partial<PlantFormData>>({
    name: plantData.name || '',
    species: plantData.species || '',
    lightLevel: plantData.lightLevel || 'medium',
    wateringFrequency: plantData.wateringFrequency || 3,
    location: plantData.location || '',
    temperatureMin: plantData.temperatureMin || undefined,
    temperatureMax: plantData.temperatureMax || undefined,
    airHumidity: plantData.airHumidity || undefined,
    soilHumidity: plantData.soilHumidity || undefined,
  });

  const [errors, setErrors] = useState<Record<string, string>>({});

  const handleChange = (field: keyof PlantFormData, value: any) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
    // Clear error for this field
    setErrors((prev) => ({
      ...prev,
      [field]: '',
    }));
  };

  const validateForm = () => {
    const newErrors: Record<string, string> = {};

    if (!formData.name?.trim()) {
      newErrors.name = 'Plant name is required';
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return false;
    }
    return true;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validateForm()) {
      onSubmit(formData);
    }
  };

  const inputClasses = (hasError: boolean) =>
    `w-full rounded-lg px-4 py-2 text-sm transition-colors ${
      darkMode
        ? `bg-neutral-800 text-white placeholder-neutral-500 border ${
            hasError ? 'border-red-500' : 'border-neutral-700'
          } focus:border-secondary focus:outline-none`
        : `bg-neutral-100 text-neutral-900 placeholder-neutral-400 border ${
            hasError ? 'border-red-500' : 'border-neutral-200'
          } focus:border-secondary focus:outline-none`
    }`;

  const labelClasses = `text-sm font-medium ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`;

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Plant Name */}
      <div>
        <label htmlFor="name" className={labelClasses}>
          Plant Name *
        </label>
        <input
          id="name"
          type="text"
          placeholder="e.g., My Monstera"
          value={formData.name || ''}
          onChange={(e) => handleChange('name', e.target.value)}
          className={inputClasses(!!errors.name)}
        />
        {errors.name && <p className="mt-1 text-xs text-red-500">{errors.name}</p>}
      </div>

      {/* Species */}
      <div>
        <label htmlFor="species" className={labelClasses}>
          Species (optional)
        </label>
        <input
          id="species"
          type="text"
          placeholder="e.g., Monstera Deliciosa"
          value={formData.species || ''}
          onChange={(e) => handleChange('species', e.target.value)}
          className={inputClasses(false)}
        />
      </div>

      {/* Location */}
      <div>
        <label htmlFor="location" className={labelClasses}>
          Location (optional)
        </label>
        <input
          id="location"
          type="text"
          placeholder="e.g., Living Room"
          value={formData.location || ''}
          onChange={(e) => handleChange('location', e.target.value)}
          className={inputClasses(false)}
        />
      </div>

      {/* Light Level */}
      <div>
        <label htmlFor="lightLevel" className={labelClasses}>
          Light Level
        </label>
        <select
          id="lightLevel"
          value={formData.lightLevel || 'medium'}
          onChange={(e) =>
            handleChange(
              'lightLevel',
              e.target.value as 'low' | 'medium' | 'high'
            )
          }
          className={inputClasses(false)}
        >
          <option value="low">🌑 Low Light</option>
          <option value="medium">☀️ Medium Light</option>
          <option value="high">🌞 Bright Light</option>
        </select>
      </div>

      {/* Watering Frequency */}
      <div>
        <label htmlFor="watering" className={labelClasses}>
          Water Every (days)
        </label>
        <input
          id="watering"
          type="number"
          min="1"
          max="30"
          value={formData.wateringFrequency || 3}
          onChange={(e) => handleChange('wateringFrequency', parseInt(e.target.value))}
          className={inputClasses(false)}
        />
      </div>

      {/* Custom Parameters Section */}
      <div className={`rounded-lg border p-4 ${darkMode ? 'border-neutral-700 bg-neutral-800/50' : 'border-neutral-200 bg-neutral-50'}`}>
        <h3 className={`mb-3 text-sm font-semibold ${darkMode ? 'text-neutral-200' : 'text-neutral-800'}`}>
          Custom Care Parameters (Optional)
        </h3>

        {/* Temperature Range */}
        <div className="grid grid-cols-2 gap-3 mb-3">
          <div>
            <label htmlFor="tempMin" className={labelClasses}>
              Min Temp (°C)
            </label>
            <input
              id="tempMin"
              type="number"
              min="-10"
              max="50"
              value={formData.temperatureMin || ''}
              onChange={(e) => handleChange('temperatureMin', e.target.value ? parseInt(e.target.value) : undefined)}
              placeholder="e.g., 15"
              className={inputClasses(false)}
            />
          </div>
          <div>
            <label htmlFor="tempMax" className={labelClasses}>
              Max Temp (°C)
            </label>
            <input
              id="tempMax"
              type="number"
              min="-10"
              max="50"
              value={formData.temperatureMax || ''}
              onChange={(e) => handleChange('temperatureMax', e.target.value ? parseInt(e.target.value) : undefined)}
              placeholder="e.g., 26"
              className={inputClasses(false)}
            />
          </div>
        </div>

        {/* Humidity Levels */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label htmlFor="airHumidity" className={labelClasses}>
              Air Humidity
            </label>
            <select
              id="airHumidity"
              value={formData.airHumidity || ''}
              onChange={(e) =>
                handleChange(
                  'airHumidity',
                  e.target.value ? (e.target.value as 'low' | 'medium' | 'high') : undefined
                )
              }
              className={inputClasses(false)}
            >
              <option value="">Not specified</option>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </div>
          <div>
            <label htmlFor="soilHumidity" className={labelClasses}>
              Soil Humidity
            </label>
            <select
              id="soilHumidity"
              value={formData.soilHumidity || ''}
              onChange={(e) =>
                handleChange(
                  'soilHumidity',
                  e.target.value ? (e.target.value as 'low' | 'medium' | 'high') : undefined
                )
              }
              className={inputClasses(false)}
            >
              <option value="">Not specified</option>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </div>
        </div>
      </div>

      {/* Buttons */}
      <div className="flex gap-3 pt-4">
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          type="button"
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
          type="submit"
          className="flex-1 rounded-lg bg-secondary px-4 py-2 font-medium text-white transition-colors hover:opacity-90"
        >
          Continue
        </motion.button>
      </div>
    </form>
  );
}
