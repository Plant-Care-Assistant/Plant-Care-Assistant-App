'use client';

import { Droplet, Wind, Sprout, Scissors, RotateCw, Eye, MoreHorizontal } from 'lucide-react';
import { CareType, DailyCare } from '@/types';

interface WeeklyCareProps {
  /** Fixed 7-day strip from backend, oldest first, last entry = today. */
  daily: DailyCare[];
  /** Distinct days with ≥1 care event this week (for the header badge). */
  activeDays: number;
  darkMode?: boolean;
}

const CARE_META: Record<CareType, { Icon: typeof Droplet; color: string; label: string }> = {
  water: { Icon: Droplet, color: 'bg-accent text-white', label: 'Watered' },
  mist: { Icon: Wind, color: 'bg-sky-400 text-white', label: 'Misted' },
  fertilize: { Icon: Sprout, color: 'bg-secondary text-white', label: 'Fertilized' },
  prune: { Icon: Scissors, color: 'bg-amber-500 text-white', label: 'Pruned' },
  rotate: { Icon: RotateCw, color: 'bg-violet-500 text-white', label: 'Rotated' },
  inspect: { Icon: Eye, color: 'bg-teal-500 text-white', label: 'Inspected' },
  other: { Icon: MoreHorizontal, color: 'bg-neutral-500 text-white', label: 'Other care' },
};

const SHORT_DAY = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];

function dayLabel(isoDate: string): string {
  // Parse as local date (avoid the UTC midnight shift when the string lacks a TZ)
  const [y, m, d] = isoDate.split('-').map(Number);
  return SHORT_DAY[new Date(y, m - 1, d).getDay()];
}

interface DayColumnProps {
  day: DailyCare;
  isToday: boolean;
  darkMode: boolean;
}

const DayColumn: React.FC<DayColumnProps> = ({ day, isToday, darkMode }) => {
  const types = day.types;
  const empty = types.length === 0;

  return (
    <div
      className={`flex flex-col items-center gap-2 p-2 sm:p-3 rounded-2xl min-h-[88px] ${
        isToday
          ? 'bg-accent/15 ring-2 ring-accent'
          : darkMode
            ? 'bg-neutral-700'
            : 'bg-neutral-100'
      }`}
    >
      <span
        className={`text-xs font-medium ${
          isToday
            ? darkMode
              ? 'text-accent'
              : 'text-accent'
            : darkMode
              ? 'text-neutral-400'
              : 'text-neutral-600'
        }`}
      >
        {dayLabel(day.date)}
      </span>

      {empty ? (
        <div
          className={`w-6 sm:w-7 h-6 sm:h-7 rounded-full ${
            darkMode ? 'bg-neutral-600' : 'bg-neutral-300'
          }`}
        />
      ) : (
        <div className="flex flex-wrap items-center justify-center gap-1 max-w-[64px]">
          {types.map((t) => {
            const meta = CARE_META[t];
            const Icon = meta.Icon;
            return (
              <div
                key={t}
                title={meta.label}
                className={`w-5 h-5 sm:w-6 sm:h-6 rounded-full flex items-center justify-center ${meta.color}`}
              >
                <Icon size={12} strokeWidth={3} />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export const WeeklyCare: React.FC<WeeklyCareProps> = ({ daily, activeDays, darkMode = false }) => {
  // The backend strip is oldest-first; today is always the last entry.
  const todayIsoDate = daily.length ? daily[daily.length - 1].date : null;

  return (
    <div className={`rounded-3xl shadow-lg p-4 sm:p-6 ${darkMode ? 'bg-neutral-800' : 'bg-white'}`}>
      <div className="flex items-center justify-between mb-4 sm:mb-6">
        <span
          className={`text-base sm:text-lg font-semibold ${
            darkMode ? 'text-white' : 'text-neutral-900'
          }`}
        >
          Weekly Care
        </span>
        <span
          className={`text-xs sm:text-sm font-medium px-3 py-1 rounded-full ${
            darkMode
              ? 'bg-neutral-700 text-neutral-300'
              : 'bg-secondary/20 text-secondary'
          }`}
        >
          {activeDays}/7 days
        </span>
      </div>
      <div className="grid grid-cols-7 gap-1 sm:gap-2">
        {daily.map((d) => (
          <DayColumn
            key={d.date}
            day={d}
            isToday={d.date === todayIsoDate}
            darkMode={darkMode}
          />
        ))}
      </div>
    </div>
  );
};
