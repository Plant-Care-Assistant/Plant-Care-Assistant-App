'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';
import { ScanCameraStep } from './ScanCameraStep';
import { ScanResultsStep } from './ScanResultsStep';
import { ScanConfirmStep } from './ScanConfirmStep';

export interface ScanPlantData {
  imageUrl: string;
  name: string;
  species?: string;
  lightLevel: 'low' | 'medium' | 'high';
  wateringFrequency: number; // days
  location?: string;
  confidence?: number; // AI confidence score 0-100
  aiIdentified?: boolean;
}

interface ScanCameraModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAddToCollection: (plant: ScanPlantData) => void;
  darkMode: boolean;
}

type Step = 'camera' | 'results' | 'confirm';

export function ScanCameraModal({
  isOpen,
  onClose,
  onAddToCollection,
  darkMode,
}: ScanCameraModalProps) {
  const [currentStep, setCurrentStep] = useState<Step>('camera');
  const [plantData, setPlantData] = useState<Partial<ScanPlantData>>({
    lightLevel: 'medium',
    wateringFrequency: 3,
  });

  const handleImageCaptured = (imageUrl: string) => {
    setPlantData((prev) => ({ ...prev, imageUrl }));
    
    // Simulate AI identification
    setTimeout(() => {
      // Mock AI result - in production, this would call your AI API
      const mockResult = {
        name: 'Monstera',
        species: 'Monstera Deliciosa',
        confidence: 87,
        aiIdentified: true,
        lightLevel: 'medium' as const,
        wateringFrequency: 7,
      };
      
      setPlantData((prev) => ({ ...prev, ...mockResult }));
      setCurrentStep('results');
    }, 1500); // Simulate processing time
  };

  const handleResultsSubmit = (details: Partial<ScanPlantData>) => {
    setPlantData((prev) => ({ ...prev, ...details }));
    setCurrentStep('confirm');
  };

  const handleConfirm = () => {
    onAddToCollection(plantData as ScanPlantData);
    // Reset for next use
    setCurrentStep('camera');
    setPlantData({ lightLevel: 'medium', wateringFrequency: 3 });
    onClose();
  };

  const handleViewOnly = () => {
    // View results without adding to collection
    setCurrentStep('camera');
    setPlantData({ lightLevel: 'medium', wateringFrequency: 3 });
    onClose();
  };

  const handleBack = () => {
    if (currentStep === 'results') {
      setCurrentStep('camera');
    } else if (currentStep === 'confirm') {
      setCurrentStep('results');
    }
  };

  const handleCloseModal = () => {
    setCurrentStep('camera');
    setPlantData({ lightLevel: 'medium', wateringFrequency: 3 });
    onClose();
  };

  const steps = ['camera', 'results', 'confirm'] as const;
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
            onClick={handleCloseModal}
            className="fixed inset-0 z-40 bg-black/50"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className={`fixed left-1/2 top-1/2 z-50 w-full max-w-md max-h-[90vh] -translate-x-1/2 -translate-y-1/2 overflow-hidden rounded-3xl flex flex-col ${
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
                  Identify Plant
                </h2>
                <p className={`text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-500'}`}>
                  Step {currentStepIndex + 1} of {steps.length}
                </p>
              </div>
              <button
                onClick={handleCloseModal}
                className={`rounded-lg p-2 transition-colors ${
                  darkMode
                    ? 'hover:bg-neutral-800'
                    : 'hover:bg-neutral-200'
                }`}
                aria-label="Close modal"
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
            <div className="overflow-y-auto overflow-x-hidden p-6 flex-1">
              <AnimatePresence mode="wait">
                {currentStep === 'camera' && (
                  <motion.div
                    key="camera"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                  >
                    <ScanCameraStep
                      onImageCaptured={handleImageCaptured}
                      darkMode={darkMode}
                    />
                  </motion.div>
                )}

                {currentStep === 'results' && (
                  <motion.div
                    key="results"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                  >
                    <ScanResultsStep
                      plantData={plantData}
                      onSubmit={handleResultsSubmit}
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
                    <ScanConfirmStep
                      plantData={plantData as ScanPlantData}
                      onConfirm={handleConfirm}
                      onViewOnly={handleViewOnly}
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
