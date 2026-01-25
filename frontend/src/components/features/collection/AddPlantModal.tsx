'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';
import { AddPlantScanStep } from './AddPlantScanStep';
import { AddPlantDetailsStep } from './AddPlantDetailsStep';
import { AddPlantConfirmStep } from './AddPlantConfirmStep';

export interface PlantFormData {
  imageUrl: string;
  name: string;
  species: string;
  lightLevel: 'low' | 'medium' | 'high';
  wateringFrequency: number; // days
  location: string;
  // Custom plant parameters
  temperatureMin?: number;
  temperatureMax?: number;
  airHumidity?: 'low' | 'medium' | 'high';
  soilHumidity?: 'low' | 'medium' | 'high';
}

interface AddPlantModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (plant: PlantFormData) => void;
  darkMode: boolean;
}

type Step = 'scan' | 'details' | 'confirm';

export function AddPlantModal({
  isOpen,
  onClose,
  onSave,
  darkMode,
}: AddPlantModalProps) {
  const [currentStep, setCurrentStep] = useState<Step>('scan');
  const [plantData, setPlantData] = useState<Partial<PlantFormData>>({
    lightLevel: 'medium',
    wateringFrequency: 3,
  });

  const handleImageSelected = (imageUrl: string) => {
    setPlantData((prev) => ({ ...prev, imageUrl }));
    setCurrentStep('details');
  };

  const handleDetailsSubmit = (details: Partial<PlantFormData>) => {
    setPlantData((prev) => ({ ...prev, ...details }));
    setCurrentStep('confirm');
  };

  const handleConfirm = () => {
    onSave(plantData as PlantFormData);
    // Reset for next use
    setCurrentStep('scan');
    setPlantData({ lightLevel: 'medium', wateringFrequency: 3 });
    onClose();
  };

  const handleBack = () => {
    if (currentStep === 'details') {
      setCurrentStep('scan');
    } else if (currentStep === 'confirm') {
      setCurrentStep('details');
    }
  };

  const steps = ['scan', 'details', 'confirm'] as const;
  const currentStepIndex = steps.indexOf(currentStep);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-40 bg-black/50"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className={`fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2 overflow-hidden rounded-3xl ${
              darkMode ? 'bg-neutral-900' : 'bg-neutral-50'
            }`}
          >
            {/* Header */}
            <div
              className={`flex items-center justify-between border-b px-6 py-4 ${
                darkMode ? 'border-neutral-800' : 'border-neutral-200'
              }`}
            >
              <div>
                <h2 className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
                  Add New Plant
                </h2>
                <p className={`text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-500'}`}>
                  Step {currentStepIndex + 1} of {steps.length}
                </p>
              </div>
              <button
                onClick={onClose}
                className={`rounded-lg p-2 transition-colors ${
                  darkMode
                    ? 'hover:bg-neutral-800'
                    : 'hover:bg-neutral-200'
                }`}
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Progress Bar */}
            <div className="h-1 bg-neutral-200 dark:bg-neutral-800">
              <motion.div
                className="h-full bg-secondary"
                initial={{ width: 0 }}
                animate={{ width: `${((currentStepIndex + 1) / steps.length) * 100}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>

            {/* Content */}
            <div className="overflow-hidden p-6">
              <AnimatePresence mode="wait">
                {currentStep === 'scan' && (
                  <motion.div
                    key="scan"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                  >
                    <AddPlantScanStep
                      onImageSelected={handleImageSelected}
                      darkMode={darkMode}
                    />
                  </motion.div>
                )}

                {currentStep === 'details' && (
                  <motion.div
                    key="details"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                  >
                    <AddPlantDetailsStep
                      plantData={plantData}
                      onSubmit={handleDetailsSubmit}
                      onBack={handleBack}
                      darkMode={darkMode}
                    />
                  </motion.div>
                )}

                {currentStep === 'confirm' && (
                  <motion.div
                    key="confirm"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                  >
                    <AddPlantConfirmStep
                      plantData={plantData as PlantFormData}
                      onConfirm={handleConfirm}
                      onBack={handleBack}
                      darkMode={darkMode}
                    />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
