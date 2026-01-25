'use client';

import { useState } from 'react';
import { ScanHeader } from '@/components/features/scan/ScanHeader';
import { ScanFrame } from '@/components/features/scan/ScanFrame';
import { ScanActions } from '@/components/features/scan/ScanActions';
import { ScanCameraModal, type ScanPlantData } from '@/components/features/scan';
import { Upload } from 'lucide-react';
import type { Plant } from '@/lib/utils/plantFilters';

export interface ScanScreenProps {
  darkMode: boolean;
  plants: Plant[];
  onPlantsChange: (plants: Plant[]) => void;
}

export function ScanScreen({ darkMode, plants, onPlantsChange }: ScanScreenProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleUpload = () => {
    setIsModalOpen(true);
  };

  const handleCamera = () => {
    setIsModalOpen(true);
  };

  const handleAddToCollection = (plant: ScanPlantData) => {
    // Convert ScanPlantData to Plant format (match Plant interface exactly)
    const newPlant: Plant = {
      id: Date.now(),
      name: plant.name || 'Unknown Plant',
      species: plant.species || '',
      health: 'healthy', // Default for new plants
      lightLevel: plant.lightLevel || 'medium',
      imageUrl: plant.imageUrl || '',
      lastWatered: 'today', // Keep as string for UI compatibility
      location: plant.location,
      wateringFrequency: plant.wateringFrequency,
      aiIdentified: plant.aiIdentified,
      confidence: plant.confidence,
    };

    // Add plant to collection
    const updatedPlants = [...plants, newPlant];
    onPlantsChange(updatedPlants);

    console.log('✅ Plant added to collection:', newPlant);
  };

  return (
    <div className={`min-h-screen pb-24 lg:pb-8 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
      {/* Mobile layout - optimized for touch */}
      <div className="lg:hidden">
        <div className="px-4 pt-4 pb-6">
          {/* Header with XP reward */}
          <ScanHeader xpReward={50} darkMode={darkMode} />

          {/* Main scan area */}
          <div className="flex flex-col">
            <ScanFrame darkMode={darkMode} />
            <ScanActions 
              onUpload={handleUpload}
              onCamera={handleCamera}
              darkMode={darkMode}
            />
          </div>
        </div>
      </div>

      {/* Desktop layout - optimized for mouse interaction */}
      <div className="hidden lg:block">
        <div className="max-w-6xl mx-auto px-6 py-8">
          {/* Header centered */}
          <div className="flex justify-center mb-8">
            <ScanHeader xpReward={50} darkMode={darkMode} />
          </div>

          <div className="grid grid-cols-1 gap-8 items-start justify-items-center">
            {/* Drag & drop area - centered */}
            <div className="flex flex-col justify-center w-full max-w-2xl">
              <div
                className={`w-full rounded-2xl border-2 border-dashed ${
                  darkMode
                    ? 'border-neutral-700 bg-neutral-800/30 hover:border-secondary/50 hover:bg-neutral-800/50'
                    : 'border-neutral-300 bg-neutral-50/50 hover:border-secondary/50 hover:bg-secondary/5'
                } p-12 text-center cursor-pointer transition-all duration-300`}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  handleUpload();
                }}
                onClick={handleUpload}
              >
                <div className="flex flex-col items-center gap-4">
                  <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
                    darkMode ? 'bg-neutral-700/50' : 'bg-neutral-200/50'
                  }`}>
                    <Upload size={28} className={darkMode ? 'text-neutral-400' : 'text-neutral-600'} />
                  </div>
                  <div>
                    <p className={`text-lg font-semibold mb-1 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                      Drag and drop an image here
                    </p>
                    <p className={`text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
                      or click to select from your files
                    </p>
                  </div>
                  <div className={`mt-2 text-xs ${darkMode ? 'text-neutral-500' : 'text-neutral-500'}`}>
                    <p>Supported formats: JPG, PNG, HEIC</p>
                    <p className="mt-1">Maximum size: 10MB</p>
                  </div>
                </div>
              </div>

              {/* Additional tips */}
              <div className={`mt-6 p-4 rounded-xl ${
                darkMode ? 'bg-neutral-800/50' : 'bg-neutral-100/50'
              }`}>
                <p className={`text-sm font-medium mb-2 ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
                  Tips for best results:
                </p>
                <ul className={`text-xs space-y-1 ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
                  <li>• Ensure good lighting on the plant</li>
                  <li>• Center the plant in the frame</li>
                  <li>• Include leaves and distinctive features</li>
                  <li>• Avoid blurry or distant photos</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Scan Camera Modal */}
      <ScanCameraModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onAddToCollection={handleAddToCollection}
        darkMode={darkMode}
      />
    </div>
  );
}
