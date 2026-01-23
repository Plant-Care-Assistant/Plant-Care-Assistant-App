import { ProfileCard } from './ProfileCard';
import { Award, Flame, TrendingUp } from 'lucide-react';

interface StatsGridProps {
  level: number;
  dayStreak: number;
  totalXP: number;
  darkMode?: boolean;
}

export function StatsGrid({ level, dayStreak, totalXP, darkMode = false }: StatsGridProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4">
      <ProfileCard
        icon={<Award size={28} />}
        value={level}
        label="Level"
        bgColor="bg-secondary"
        darkMode={darkMode}
      />
      <ProfileCard
        icon={<Flame size={28} />}
        value={dayStreak}
        label="Day Streak"
        bgColor="bg-accent2"
        darkMode={darkMode}
      />
      <ProfileCard
        icon={<TrendingUp size={28} />}
        value={totalXP}
        label="Total XP"
        bgColor="bg-primary"
        darkMode={darkMode}
      />
    </div>
  );
}
