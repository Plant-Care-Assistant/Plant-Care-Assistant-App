'use client';

import { Check } from 'lucide-react';

interface WeeklyCareProps {
  totalDays: number; // e.g., 7
  activeDays: number; // e.g., 3
  darkMode?: boolean;
}

const days = ['M', 'T', 'W', 'T', 'F', 'S', 'S'];

interface DayIndicatorProps {
  day: string;
  isCompleted: boolean;
  isCurrent: boolean;
  darkMode?: boolean;
}

const DayIndicator: React.FC<DayIndicatorProps> = ({ day, isCompleted, isCurrent, darkMode = false }) => {
  return (
    <div className={`flex flex-col items-center gap-2 p-2 sm:p-3 rounded-2xl ${
      isCurrent
        ? 'bg-accent'
        : darkMode
          ? 'bg-neutral-700'
          : 'bg-neutral-100'
    }`}>
      <span className={`text-xs font-medium ${
        isCurrent
          ? 'text-white'
          : darkMode
            ? 'text-neutral-400'
            : 'text-neutral-600'
      }`}>
        {day}
      </span>
      {isCurrent ? (
        // Current/active day - white checkmark with subtle white circle on blue background
        <div className="w-6 sm:w-7 h-6 sm:h-7 rounded-full bg-white/40 flex items-center justify-center">
          <Check className="text-white" size={16} strokeWidth={3} />
        </div>
      ) : isCompleted ? (
        // Completed day - green circle with checkmark
        <div className="w-6 sm:w-7 h-6 sm:h-7 rounded-full bg-secondary flex items-center justify-center">
          <Check className="text-white" size={14} strokeWidth={3} />
        </div>
      ) : (
        // Future day - darker gray circle
        <div className={`w-6 sm:w-7 h-6 sm:h-7 rounded-full ${darkMode ? 'bg-neutral-600' : 'bg-neutral-300'}`}></div>
      )}
    </div>
  );
};

export const WeeklyCare: React.FC<WeeklyCareProps> = ({ totalDays, activeDays, darkMode = false }) => {
  return (
    <div className={`rounded-3xl shadow-lg p-4 sm:p-6 ${darkMode ? 'bg-neutral-800' : 'bg-white'}`}>
      <div className="flex items-center justify-between mb-4 sm:mb-6">
        <span className={`text-base sm:text-lg font-semibold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
          Weekly Care
        </span>
        <span className={`text-xs sm:text-sm font-medium px-3 py-1 rounded-full ${
          darkMode 
            ? 'bg-neutral-700 text-neutral-300' 
            : 'bg-secondary/20 text-secondary'
        }`}>
          {activeDays}/{totalDays} days
        </span>
      </div>
      <div className="grid grid-cols-7 gap-1 sm:gap-2">
        {days.map((d, i) => (
          <DayIndicator 
            key={i} 
            day={d} 
            isCompleted={i < activeDays && i !== activeDays - 1}
            isCurrent={i === activeDays - 1}
            darkMode={darkMode}
          />
        ))}
      </div>
    </div>
  );
};
