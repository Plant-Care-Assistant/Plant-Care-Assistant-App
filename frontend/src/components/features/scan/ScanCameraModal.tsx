'use client';

import { useState } from 'react';
import { AIAPI, PlantPrediction } from '../../../lib/apiClient';
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
  confidence?: number; // Confidence score 0-100
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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Accepts both imageUrl (data URL) and imageFile (File)
  const handleImageCaptured = async (imageUrl: string, imageFile?: File | Blob) => {
    setPlantData((prev) => ({ ...prev, imageUrl }));
    setError(null);
    setLoading(true);
    try {
      if (imageFile) {
        let fileToSend: File;
        if (imageFile instanceof File) {
          fileToSend = imageFile;
        } else {
          // Convert Blob to File (default name 'capture.jpg')
          fileToSend = new File([imageFile], 'capture.jpg', { type: imageFile.type || 'image/jpeg' });
        }
        const prediction: PlantPrediction = await AIAPI.identifyPlant(fileToSend);
        // Pick top 5 predictions, use the top one for prefill
        const top = prediction.predictions[0];
        setPlantData((prev) => ({
          ...prev,
          name: top?.class_name || '',
          species: top?.class_name || '',
          confidence: top?.confidence,
          aiIdentified: !!top,
          // Keep imageUrl, lightLevel, wateringFrequency
        }));
      } else {
        // No file, fallback to manual entry
        setPlantData((prev) => ({ ...prev, aiIdentified: false }));
      }
      setCurrentStep('results');
    } catch (err: any) {
      setError(err?.message || 'Failed to identify plant. Please try again.');
    } finally {
      setLoading(false);
    }
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
              {/* Loading/Progress/Error UI */}
              {loading && (
                <div className="flex flex-col items-center justify-center h-full gap-4">
                  <div className="w-12 h-12 border-4 border-secondary border-t-transparent rounded-full animate-spin" />
                  <div className={`text-lg font-medium ${darkMode ? 'text-white' : 'text-neutral-900'}`}>Identifying plant...</div>
                </div>
              )}
              {error && (
                <div className="flex flex-col items-center justify-center h-full gap-4">
                  <div className="text-red-500 font-semibold">{error}</div>
                  <button
                    className="rounded-xl bg-secondary text-white px-4 py-2 mt-2"
                    onClick={() => {
                      setError(null);
                      setCurrentStep('camera');
                    }}
                  >Try Again</button>
                </div>
              )}
              {!loading && !error && (
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
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
