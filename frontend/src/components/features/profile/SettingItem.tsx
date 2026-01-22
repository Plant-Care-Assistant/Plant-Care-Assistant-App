'use client';

import { ReactNode, useState } from 'react';

interface SettingItemProps {
  icon: ReactNode;
  title: string;
  description: string;
  bgColor: string;
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
  darkMode?: boolean;
}

export const SettingItem: React.FC<SettingItemProps> = ({
  icon,
  title,
  description,
  bgColor,
  enabled,
  onToggle,
  darkMode = false,
}) => {
  return (
    <div className="flex items-center justify-between gap-4">
      {/* Icon and Text */}
      <div className="flex items-center gap-4 flex-1">
        {/* Icon Badge */}
        <div className={`w-12 h-12 sm:w-14 sm:h-14 rounded-xl ${bgColor} flex items-center justify-center flex-shrink-0`}>
          {icon}
        </div>

        {/* Title and Description */}
        <div className="flex-1 min-w-0">
          <h3 className={`font-semibold text-sm sm:text-base ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
            {title}
          </h3>
          <p className={`text-xs sm:text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
            {description}
          </p>
        </div>
      </div>

      {/* Toggle Switch */}
      <button
        onClick={() => onToggle(!enabled)}
        className={`relative w-12 h-7 sm:w-14 sm:h-8 rounded-full transition-colors flex-shrink-0 ${
          enabled ? 'bg-secondary' : (darkMode ? 'bg-neutral-600' : 'bg-neutral-300')
        }`}
        aria-pressed={enabled}
        aria-label={`Toggle ${title}`}
      >
        <div
          className={`absolute top-1 left-1 w-5 h-5 sm:w-6 sm:h-6 bg-white rounded-full transition-transform ${
            enabled ? 'translate-x-5 sm:translate-x-6' : 'translate-x-0'
          }`}
        />
      </button>
    </div>
  );
};
