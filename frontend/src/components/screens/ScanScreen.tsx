'use client';

import { useState } from 'react';
import { ScanHeader } from '@/components/features/scan/ScanHeader';
import { ScanFrame } from '@/components/features/scan/ScanFrame';
import { ScanActions } from '@/components/features/scan/ScanActions';
import { ScanCameraModal, type ScanPlantData } from '@/components/features/scan';
import { Upload } from 'lucide-react';
import { useAddPlantMutation } from '@/hooks/usePlants';
import { useTheme } from '@/providers';

export function ScanScreen() {
  const { theme } = useTheme();
  const darkMode = theme === 'dark';
  const [isModalOpen, setIsModalOpen] = useState(false);
  const addPlantMutation = useAddPlantMutation();

  const handleUpload = () => {
    setIsModalOpen(true);
  };

  const handleCamera = () => {
    setIsModalOpen(true);
  };

  const handleAddToCollection = (plant: ScanPlantData) => {
    const hasHealthVerdict = plant.healthLabel != null;
    // AI returns 0..1 fraction; store as 0..100 percent.
    const healthConfPct =
      hasHealthVerdict && plant.healthConfidence != null
        ? Math.round(plant.healthConfidence * 100)
        : null;
    addPlantMutation.mutate({
      custom_name: plant.name || 'Unknown Plant',
      note: plant.species || null,
      plant_catalog_id: plant.catalogId ?? null,
      imageUrl: plant.imageUrl,
      last_health_label: plant.healthLabel ?? null,
      last_health_confidence: healthConfPct,
      last_health_check_at: hasHealthVerdict ? new Date().toISOString() : null,
      last_diseases: plant.diseases ?? null,
    });
  };

  return (
    <div className={`min-h-screen pb-24 lg:pb-8 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
      <div className="lg:hidden">
        <div className="px-4 pt-4 pb-6">
          <ScanHeader xpReward={50} darkMode={darkMode} />

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

      <div className="hidden lg:block">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <div className="flex justify-center mb-8">
            <ScanHeader xpReward={50} darkMode={darkMode} />
          </div>

          <div className="grid grid-cols-1 gap-8 items-start justify-items-center">
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

      <ScanCameraModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onAddToCollection={handleAddToCollection}
        darkMode={darkMode}
      />
    </div>
  );
}
