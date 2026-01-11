'use client';

import { Flame } from 'lucide-react';

interface DayStreakCardProps {
  days: number;
}

export const DayStreakCard: React.FC<DayStreakCardProps> = ({ days }) => {
  return (
    <div className="rounded-3xl shadow-lg bg-accent2 px-4 sm:px-6 py-4 sm:py-5">
      {/* Content */}
      <div className="flex items-center gap-3 sm:gap-5">
        <div className="w-14 sm:w-16 h-14 sm:h-16 rounded-2xl bg-white/25 flex items-center justify-center flex-shrink-0">
          <Flame className="text-white" size={24} fill="white" />
        </div>
        <div className="flex flex-col justify-center">
          <span className="text-3xl sm:text-4xl font-bold text-white leading-none">{days}</span>
          <span className="text-sm sm:text-base text-white mt-1">Day Streak</span>
        </div>
      </div>
    </div>
  );
}
