import { Flame } from 'lucide-react';

interface StreakCardProps {
  streak: number;
  darkMode: boolean;
}

export function StreakCard({ streak, darkMode }: StreakCardProps) {
  return (
    <div className={`p-5 rounded-3xl ${darkMode ? 'bg-accent2' : 'bg-accent2'} shadow-md flex flex-col items-center justify-center`}>
      <Flame className="w-7 h-7 text-white mb-2" />
      <p className="text-3xl font-bold text-white">{streak}</p>
      <p className="text-xs text-white/80 text-center">week streak</p>
    </div>
  );
}
