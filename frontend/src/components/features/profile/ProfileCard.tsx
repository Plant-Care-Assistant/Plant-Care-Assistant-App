'use client';

import { ReactNode } from 'react';

interface ProfileCardProps {
  icon: ReactNode;
  value: string | number;
  label: string;
  bgColor: string;
  darkMode?: boolean;
}

export const ProfileCard: React.FC<ProfileCardProps> = ({ 
  icon, 
  value, 
  label, 
  bgColor,
  darkMode = false 
}) => {
  return (
    <div className={`rounded-3xl shadow-lg p-6 sm:p-8 flex flex-col items-center text-center ${darkMode ? 'bg-neutral-800' : 'bg-white'}`}>
      {/* Icon Container */}
      <div className={`w-14 h-14 sm:w-16 sm:h-16 rounded-2xl flex items-center justify-center mb-4 ${bgColor}`}>
        <span className="text-white text-2xl sm:text-3xl">{icon}</span>
      </div>
      
      {/* Value */}
      <div className={`text-3xl sm:text-4xl font-bold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
        {value}
      </div>
      
      {/* Label */}
      <div className={`text-sm sm:text-base mt-2 ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
        {label}
      </div>
    </div>
  );
};
