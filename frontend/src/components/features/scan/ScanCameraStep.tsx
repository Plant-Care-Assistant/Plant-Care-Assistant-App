'use client';

import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { motion } from 'framer-motion';
import { Camera, Image as ImageIcon } from 'lucide-react';
import { CameraCapture } from './CameraCapture';

interface ScanCameraStepProps {
  onImageCaptured: (imageUrl: string, imageFile?: File | Blob) => void;
  darkMode: boolean;
}

export function ScanCameraStep({
  onImageCaptured,
  darkMode,
}: ScanCameraStepProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [isMounted, setIsMounted] = useState(false);

  // Ensure DOM is ready before rendering portal (SSR safety)
  useEffect(() => {
    setIsMounted(true);
  }, []);

  const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Validate file size
      if (file.size > MAX_FILE_SIZE) {
        alert(
          `File too large. Maximum size is 5MB. You selected ${(file.size / 1024 / 1024).toFixed(2)}MB`
        );
        return;
      }
      // Validate file type
      if (!file.type.startsWith("image/")) {
        alert("Please select an image file");
        return;
      }
      const reader = new FileReader();
      reader.onerror = () => {
        alert("Failed to read file");
      };
      reader.onload = (event) => {
        const result = event.target?.result as string;
        onImageCaptured(result, file);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleCameraClick = () => {
    // Open the CameraCapture component
    setIsCameraOpen(true);
  };

  const handleCameraCapture = (imageBlob: Blob) => {
    // Convert blob to data URL for backward compatibility
    const reader = new FileReader();
    reader.onload = (event) => {
      const result = event.target?.result as string;
      onImageCaptured(result, imageBlob);
    };
    reader.readAsDataURL(imageBlob);
    setIsCameraOpen(false);
  };

  const handleGalleryClick = () => {
    // Open file picker for gallery
    if (fileInputRef.current) {
      fileInputRef.current.removeAttribute('capture');
      fileInputRef.current.click();
    }
  };

  return (
    <div className="space-y-4">
      <p className={`text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
        Capture or upload a photo of your plant to identify it with AI-powered recognition.
      </p>

      {/* Open Camera Option */}
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={handleCameraClick}
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
            Open Camera
          </p>
          <p className={`text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-500'}`}>
            Take a photo with your camera
          </p>
        </div>
      </motion.button>

      {/* Choose from Gallery Option */}
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={handleGalleryClick}
        className={`w-full flex items-center gap-4 rounded-2xl border-2 p-4 transition-all ${
          darkMode
            ? 'border-neutral-700 hover:border-secondary hover:bg-neutral-800'
            : 'border-neutral-300 hover:border-secondary hover:bg-neutral-100'
        }`}
      >
        <div className={`rounded-lg p-3 ${darkMode ? 'bg-neutral-800' : 'bg-neutral-100'}`}>
          <ImageIcon className="h-6 w-6 text-secondary" />
        </div>
        <div className="text-left">
          <p className={`font-semibold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
            Choose from Gallery
          </p>
          <p className={`text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-500'}`}>
            Select an existing photo
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
      <div className={`mt-6 rounded-xl p-4 ${darkMode ? 'bg-neutral-800' : 'bg-neutral-100'}`}>
        <p className={`text-xs ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
          💡 <strong>Tip:</strong> For best results, take a clear photo with good lighting showing the plant's leaves and overall shape.
        </p>
      </div>

      {/* Camera Capture Component - rendered as portal to escape modal z-index */}
      {isMounted && typeof window !== 'undefined' && createPortal(
        <CameraCapture
          isOpen={isCameraOpen}
          onClose={() => setIsCameraOpen(false)}
          onCapture={handleCameraCapture}
        />,
        document.body
      )}
    </div>
  );
}
