'use client';

import { Upload, Camera } from 'lucide-react';
import { motion } from 'framer-motion';

interface ScanActionsProps {
  onUpload: () => void;
  onCamera: () => void;
  darkMode: boolean;
}

export const ScanActions: React.FC<ScanActionsProps> = ({ 
  onUpload, 
  onCamera, 
  darkMode 
}) => {
  return (
    <div className="w-full max-w-md mx-auto px-4">
      {/* Action buttons container */}
      <div className="flex items-center justify-center gap-4 lg:gap-6">
        {/* Upload button - secondary action */}
        <motion.button
          onClick={onUpload}
          className={`relative w-14 h-14 lg:w-16 lg:h-16 rounded-full flex items-center justify-center transition-all shadow-md ${
            darkMode
              ? 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700 border-2 border-neutral-700'
              : 'bg-white text-neutral-700 hover:bg-neutral-50 border-2 border-neutral-200'
          }`}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          title="Upload image from gallery"
        >
          <Upload size={22} strokeWidth={2.5} />
        </motion.button>

        {/* Camera button - primary action with emphasis */}
        <motion.button
          onClick={onCamera}
          className="relative w-20 h-20 lg:w-24 lg:h-24 rounded-full bg-gradient-to-br from-secondary to-secondary/90 text-white flex items-center justify-center shadow-2xl ring-4 ring-secondary/20"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          title="Take photo with camera"
        >
          {/* Pulse animation ring */}
          <motion.div
            className="absolute inset-0 rounded-full bg-secondary/30"
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.5, 0, 0.5],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
          <Camera size={36} strokeWidth={2} className="relative z-10" />
        </motion.button>

        {/* Spacer for visual balance */}
        <div className="w-14 h-14 lg:w-16 lg:h-16" />
      </div>

      {/* Instruction text */}
      <div className="mt-6 text-center">
        <p className={`text-sm font-medium ${
          darkMode ? 'text-neutral-300' : 'text-neutral-700'
        }`}>
          Tap to identify plant and earn rewards
        </p>
        <p className={`mt-1 text-xs ${
          darkMode ? 'text-neutral-500' : 'text-neutral-500'
        }`}>
          Works best with clear, well-lit photos
        </p>
      </div>
    </div>
  );
};
