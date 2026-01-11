'use client';

import Image from 'next/image';

interface ProfileHeaderProps {
  name: string;
  level: number;
  totalXP: number;
  dayStreak: number;
  achievements: string;
  darkMode?: boolean;
}

export const ProfileHeader: React.FC<ProfileHeaderProps> = ({ 
  name, 
  level, 
  totalXP, 
  dayStreak, 
  achievements, 
  darkMode = false 
}) => {
  return (
    <div className={`rounded-3xl shadow-lg p-6 sm:p-8 ${darkMode ? 'bg-neutral-800' : 'bg-white'}`}>
      {/* Header Section */}
      <div className="flex items-start gap-4 sm:gap-6 mb-8 pb-8 border-b border-neutral-200 dark:border-neutral-700">
        {/* Avatar */}
        <div className={`w-16 h-16 sm:w-20 sm:h-20 rounded-3xl flex items-center justify-center flex-shrink-0 p-3 border-2 shadow-md ${
          darkMode ? 'bg-white/10 border-neutral-600' : 'bg-white border-neutral-300'
        }`}>
          <Image
            src="/ic-8.png"
            alt="Profile avatar"
            width={56}
            height={56}
            className="object-contain"
          />
        </div>

        {/* User Info */}
        <div className="flex-1">
          <h1 className={`text-2xl sm:text-3xl font-bold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
            {name}
          </h1>
          <p className={`text-sm sm:text-base ${darkMode ? 'text-neutral-400' : 'text-neutral-600'} mt-1`}>
            Level {level} Plant Parent
          </p>
          
          {/* Level Badge */}
          <div className="mt-3 inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary text-white">
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.381-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
            </svg>
            <span className="font-semibold text-sm">Level {level}</span>
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="grid grid-cols-3 gap-2 sm:gap-4">
        <div className="flex flex-col items-center text-center">
          <div className={`text-2xl sm:text-3xl font-bold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
            {totalXP}
          </div>
          <div className={`text-xs sm:text-sm mt-2 ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
            Total XP
          </div>
        </div>
        <div className="flex flex-col items-center text-center">
          <div className={`text-2xl sm:text-3xl font-bold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
            {dayStreak}
          </div>
          <div className={`text-xs sm:text-sm mt-2 ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
            Day Streak
          </div>
        </div>
        <div className="flex flex-col items-center text-center">
          <div className={`text-2xl sm:text-3xl font-bold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
            {achievements}
          </div>
          <div className={`text-xs sm:text-sm mt-2 ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
            Achievements
          </div>
        </div>
      </div>
    </div>
  );
};
