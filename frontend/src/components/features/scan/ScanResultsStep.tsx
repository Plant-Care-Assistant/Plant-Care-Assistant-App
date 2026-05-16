'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { ChevronLeft, Sparkles, AlertCircle, CheckCircle2 } from 'lucide-react';
import Image from 'next/image';
import type { ScanPlantData } from './ScanCameraModal';
import { getDiseaseAdvice } from '@/lib/disease/advice';

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
  const [errors, setErrors] = useState<{
    name?: string;
    species?: string;
    location?: string;
  }>({});

  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitError(null);
    const newErrors: typeof errors = {};

    if (!formData.name?.trim()) {
      newErrors.name = 'Plant name is required';
    }

    setErrors(newErrors);
    if (Object.keys(newErrors).length === 0) {
      setIsSubmitting(true);
      try {
        await Promise.resolve(onSubmit(formData));
        setIsSubmitting(false);
      } catch (err: any) {
        setIsSubmitting(false);
        setSubmitError(err?.message || 'Failed to submit. Please try again.');
      }
    }
  };

  const lightLevels = [
    { value: 'low', label: 'Low Light', icon: '🌙' },
    { value: 'medium', label: 'Medium Light', icon: '☀️' },
    { value: 'high', label: 'Bright Light', icon: '🌞' },
  ] as const;

  const isAIIdentified = plantData.aiIdentified && (plantData.confidence || 0) > 50;

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-col gap-4 min-h-[80vh] pb-28"
    >
      {submitError && (
        <div className="rounded-xl bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 p-4 text-center font-medium">
          {submitError}
          <button
            type="button"
            className="ml-4 px-3 py-1 rounded bg-red-500 text-white text-sm font-semibold hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-red-400"
            onClick={() => setSubmitError(null)}
          >
            Dismiss
          </button>
        </div>
      )}
      {isAIIdentified ? (
        <div className={`rounded-xl p-4 flex items-start gap-3 ${
          darkMode ? 'bg-secondary/10 border border-secondary/20' : 'bg-secondary/10 border border-secondary/20'
        }`}>
          <Sparkles className="h-5 w-5 text-secondary flex-shrink-0 mt-0.5" />
          <div>
            <p className={`font-semibold text-sm ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
              Identified with {plantData.confidence}% confidence
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

      {plantData.healthLabel != null && (
        <div
          className={`rounded-xl p-4 flex flex-col gap-2 border ${
            plantData.healthLabel === 'diseased'
              ? 'bg-red-50 border-red-200 dark:bg-red-900/20 dark:border-red-800'
              : 'bg-green-50 border-green-200 dark:bg-green-900/20 dark:border-green-800'
          }`}
        >
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              {plantData.healthLabel === 'diseased' ? (
                <AlertCircle className="h-4 w-4 text-red-500 flex-shrink-0" />
              ) : (
                <CheckCircle2 className="h-4 w-4 text-green-500 flex-shrink-0" />
              )}
              <span
                className={`font-semibold text-sm ${
                  darkMode ? 'text-white' : 'text-neutral-900'
                }`}
              >
                {plantData.healthLabel === 'diseased'
                  ? 'Plant may be diseased'
                  : 'Plant looks healthy'}
              </span>
            </div>
            {plantData.healthConfidence != null && (
              <span
                className={`text-sm font-semibold ${
                  plantData.healthLabel === 'diseased' ? 'text-red-600 dark:text-red-300' : 'text-green-600 dark:text-green-300'
                }`}
              >
                {Math.round(plantData.healthConfidence * 100)}%
              </span>
            )}
          </div>

          {plantData.healthLabel === 'diseased' && plantData.diseases?.length ? (
            <>
              <ul className="mt-1 space-y-1">
                {plantData.diseases.slice(0, 3).map((d, i) => (
                  <li key={i} className="flex justify-between text-xs">
                    <span className={darkMode ? 'text-neutral-300' : 'text-neutral-700'}>
                      {/* Strip the training-set host prefix; fall back to plant token if no separator. */}
                      {d.condition || d.plant}
                    </span>
                    <span className="text-neutral-400 ml-2 shrink-0">
                      {Math.round(d.confidence * 100)}%
                    </span>
                  </li>
                ))}
              </ul>
              {/* Advice for the top disease only; re-scan to focus on a different one. */}
              {(() => {
                const top = plantData.diseases[0];
                const advice = getDiseaseAdvice(top.condition || top.plant);
                if (!advice) return null;
                return (
                  <div className="mt-3 pt-3 border-t border-red-200 dark:border-red-800">
                    <p className={`text-xs font-semibold mb-1 ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
                      What to do:
                    </p>
                    <ul className={`text-xs space-y-1 list-disc pl-4 ${
                      darkMode ? 'text-neutral-300' : 'text-neutral-700'
                    }`}>
                      {advice.map((tip, i) => <li key={i}>{tip}</li>)}
                    </ul>
                  </div>
                );
              })()}
            </>
          ) : null}
        </div>
      )}

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

      <div>
        <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
          Plant Name <span className="text-red-500">*</span>
        </label>
        <input
          type="text"
          required
          value={formData.name}
          onChange={(e) => {
            setFormData({ ...formData, name: e.target.value });
            if (errors.name) {
              setErrors({ ...errors, name: undefined });
            }
          }}
          placeholder="e.g., My Monstera"
          className={`w-full rounded-xl border px-4 py-3 transition-colors ${
            errors.name ? 'border-red-500' : darkMode ? 'border-neutral-700' : 'border-neutral-300'
          } ${
            darkMode
              ? 'bg-neutral-800 text-white placeholder:text-neutral-500 focus:border-secondary'
              : 'bg-white text-neutral-900 placeholder:text-neutral-400 focus:border-secondary'
          } focus:outline-none focus:ring-2 focus:ring-secondary/20`}
        />
        {errors.name && <p className="text-red-500 text-sm mt-1">{errors.name}</p>}
      </div>

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

      <div
        className="pt-3 pb-4 px-1"
      >
        <div className="flex gap-3">
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
            className="flex-1 rounded-xl bg-secondary px-4 py-3 font-medium text-white transition-colors hover:bg-secondary/90 disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Submitting...' : 'Continue to Summary'}
          </button>
        </div>
      </div>
    </form>
  );
}
