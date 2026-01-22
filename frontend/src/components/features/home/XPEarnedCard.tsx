import { Zap } from 'lucide-react';

interface XPEarnedCardProps {
  xp: number;
  darkMode: boolean;
}

export function XPEarnedCard({ xp, darkMode }: XPEarnedCardProps) {
  return (
    <div className={`p-5 rounded-3xl ${darkMode ? 'bg-primary-dark' : 'bg-primary'} shadow-md flex flex-col items-center justify-center`}>
      <Zap className="w-7 h-7 text-white mb-2" />
      <p className="text-3xl font-bold text-white">{xp}</p>
      <p className="text-xs text-white/80 text-center">XP earned</p>
    </div>
  );
}
