'use client';

import Image from 'next/image';
import { Lock } from 'lucide-react';

interface AchievementCardProps {
  imageSrc?: string;
  title: string;
  description: string;
  unlocked?: boolean;
  darkMode?: boolean;
}

export const AchievementCard: React.FC<AchievementCardProps> = ({
  imageSrc,
  title,
  description,
  unlocked = true,
  darkMode = false,
}) => {
  return (
    <div className={`rounded-3xl p-5 sm:p-6 flex flex-col items-center text-center ${
      darkMode ? 'bg-neutral-800' : 'bg-white'
    } ${!unlocked ? 'opacity-60' : ''}`}>
      {/* Icon/Illustration */}
      <div className={`w-14 h-14 sm:w-16 sm:h-16 rounded-2xl flex items-center justify-center mb-3 sm:mb-4 p-4 border-2 ${
        unlocked ? (darkMode ? 'bg-white/10 border-neutral-600' : 'bg-white border-neutral-300') : (darkMode ? 'bg-neutral-700 border-neutral-600' : 'bg-neutral-100 border-neutral-200')
      }`}>
        {unlocked && imageSrc ? (
          <Image
            src={imageSrc}
            alt={title}
            width={40}
            height={40}
            className="object-contain"
          />
        ) : (
          <Lock size={20} className={darkMode ? 'text-neutral-400' : 'text-neutral-400'} />
        )}
      </div>

      {/* Title */}
      <h3 className={`font-bold text-sm sm:text-base ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
        {title}
      </h3>

      {/* Description */}
      <p className={`text-xs sm:text-sm mt-1 ${darkMode ? 'text-neutral-400' : 'text-neutral-500'}`}>
        {description}
      </p>
    </div>
  );
};
