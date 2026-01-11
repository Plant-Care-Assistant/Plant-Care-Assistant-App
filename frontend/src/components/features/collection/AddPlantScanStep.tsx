'use client';

import { useRef } from 'react';
import { motion } from 'framer-motion';
import { Camera, Upload } from 'lucide-react';

interface AddPlantScanStepProps {
  onImageSelected: (imageUrl: string) => void;
  darkMode: boolean;
}

export function AddPlantScanStep({
  onImageSelected,
  darkMode,
}: AddPlantScanStepProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const result = event.target?.result as string;
        onImageSelected(result);
      };
      reader.readAsDataURL(file);
    }
  };

  const mockCameraCapture = () => {
    // In a real app, this would open the camera
    // For now, we'll use a placeholder or trigger file input
    console.log('Open camera');
    // Could implement actual camera API here
    // For demo, open file picker
    fileInputRef.current?.click();
  };

  return (
    <div className="space-y-4">
      <p className={`text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
        First, let's capture your plant. You can use your camera or upload an existing photo.
      </p>

      {/* Camera Option */}
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={mockCameraCapture}
        className={`w-full flex items-center gap-4 rounded-2xl border-2 p-4 transition-all ${
          darkMode
            ? 'border-neutral-700 hover:border-secondary hover:bg-neutral-800'
            : 'border-neutral-300 hover:border-secondary hover:bg-neutral-100'
        }`}
      >
        <div className={`rounded-lg p-3 ${darkMode ? 'bg-neutral-800' : 'bg-neutral-100'}`}>
          <Camera className="h-6 w-6 text-secondary" />
        </div>
        <div className="text-left">
          <p className={`font-semibold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
            Scan with Camera
          </p>
          <p className={`text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-500'}`}>
            Use your device's camera
          </p>
        </div>
      </motion.button>

      {/* Upload Option */}
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={() => fileInputRef.current?.click()}
        className={`w-full flex items-center gap-4 rounded-2xl border-2 p-4 transition-all ${
          darkMode
            ? 'border-neutral-700 hover:border-secondary hover:bg-neutral-800'
            : 'border-neutral-300 hover:border-secondary hover:bg-neutral-100'
        }`}
      >
        <div className={`rounded-lg p-3 ${darkMode ? 'bg-neutral-800' : 'bg-neutral-100'}`}>
          <Upload className="h-6 w-6 text-secondary" />
        </div>
        <div className="text-left">
          <p className={`font-semibold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
            Upload Photo
          </p>
          <p className={`text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-500'}`}>
            Choose from your gallery
          </p>
        </div>
      </motion.button>

      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        className="hidden"
      />

      {/* Info Text */}
      <div
        className={`rounded-xl border px-3 py-2 text-xs ${
          darkMode
            ? 'border-neutral-800 bg-neutral-800/50 text-neutral-400'
            : 'border-neutral-200 bg-neutral-100 text-neutral-600'
        }`}
      >
        💡 Tip: A clear photo of your whole plant helps with identification
      </div>
    </div>
  );
}
