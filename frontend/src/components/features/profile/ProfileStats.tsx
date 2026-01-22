'use client';

import { ProfileStat } from './ProfileStat';

interface ProfileStatsProps {
  totalXP: number;
  dayStreak: number;
  achievements: string; // e.g., "3/6"
  darkMode?: boolean;
}

export const ProfileStats: React.FC<ProfileStatsProps> = ({ totalXP, dayStreak, achievements, darkMode = false }) => {
  return (
    <div className="grid grid-cols-3 gap-2 sm:gap-4">
      <ProfileStat value={totalXP} label="Total XP" darkMode={darkMode} />
      <ProfileStat value={dayStreak} label="Day Streak" darkMode={darkMode} />
      <ProfileStat value={achievements} label="Achievements" darkMode={darkMode} />
    </div>
  );
};
