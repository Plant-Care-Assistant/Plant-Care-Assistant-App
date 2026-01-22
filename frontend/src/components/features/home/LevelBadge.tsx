'use client';

import { Zap } from 'lucide-react';

export interface LevelBadgeProps {
  level: number;
  darkMode: boolean;
}

export function LevelBadge({ level, darkMode }: LevelBadgeProps) {
  return (
    <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full font-semibold ${
      darkMode 
        ? 'bg-primary text-white' 
        : 'bg-accent text-white'
    }`}>
      <Zap size={18} />
      <span>Level {level}</span>
    </div>
  );
}
