'use client';

import { Sprout } from 'lucide-react';
import { motion } from 'framer-motion';

interface CollectionHeaderProps {
  plantCount: number;
  darkMode: boolean;
}

export const CollectionHeader: React.FC<CollectionHeaderProps> = ({ plantCount, darkMode }) => {
  return (
    <motion.div 
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="mb-6"
    >
      <div className={`rounded-2xl p-4 lg:p-6 border-2 border-dashed ${
        darkMode 
          ? 'border-neutral-700 bg-neutral-800/30' 
          : 'border-neutral-300 bg-neutral-50/50'
      }`}>
        <h1 className={`text-2xl lg:text-3xl font-bold mb-2 ${
          darkMode ? 'text-white' : 'text-gray-900'
        }`}>
          My Collection
        </h1>
        <div className="flex items-center gap-2">
          <Sprout size={18} className="text-secondary" />
          <p className={`text-sm font-medium ${
            darkMode ? 'text-neutral-300' : 'text-neutral-700'
          }`}>
            {plantCount} plant{plantCount !== 1 ? 's' : ''} growing strong
          </p>
        </div>
      </div>
    </motion.div>
  );
};
