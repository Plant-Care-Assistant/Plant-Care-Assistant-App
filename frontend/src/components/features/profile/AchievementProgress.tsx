'use client';

import { Bookmark } from 'lucide-react';

interface AchievementProgressProps {
  unlocked: number;
  total: number;
  darkMode?: boolean;
}

export const AchievementProgress: React.FC<AchievementProgressProps> = ({
  unlocked,
  total,
  darkMode = false,
}) => {
  const progressPercentage = (unlocked / total) * 100;

  return (
    <div className={`rounded-3xl shadow-lg p-6 sm:p-8 ${darkMode ? 'bg-neutral-800' : 'bg-white'}`}>
      {/* Header with Icon */}
      <div className="flex items-start gap-4 mb-6">
        {/* Icon Badge */}
        <div className="w-12 h-12 sm:w-14 sm:h-14 rounded-xl bg-nature flex items-center justify-center flex-shrink-0">
          <Bookmark size={24} className="text-white" />
        </div>

        {/* Content */}
        <div className="flex-1">
          <h3 className={`text-lg sm:text-xl font-bold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
            Achievement Progress
          </h3>
          <p className={`text-sm sm:text-base ${darkMode ? 'text-neutral-400' : 'text-neutral-600'} mt-1`}>
            {unlocked} of {total} unlocked
          </p>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="w-full h-3 sm:h-4 rounded-full bg-neutral-200 dark:bg-neutral-700 overflow-hidden">
        <div
          className="h-full bg-nature rounded-full transition-all duration-300"
          style={{ width: `${progressPercentage}%` }}
        />
      </div>
    </div>
  );
};
