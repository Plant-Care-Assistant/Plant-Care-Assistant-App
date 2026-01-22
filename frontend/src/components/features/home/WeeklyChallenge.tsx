import { Medal } from 'lucide-react';

interface WeeklyChallengeProps {
  current: number;
  total: number;
  description: string;
  darkMode: boolean;
}

export function WeeklyChallenge({ current, total, description, darkMode }: WeeklyChallengeProps) {
  const percentage = Math.round((current / total) * 100);

  return (
    <div className={`p-5 rounded-3xl ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-md mb-5`}>
      <div className="flex items-start gap-3 mb-4">
        <div className="w-12 h-12 rounded-2xl bg-nature/20 flex items-center justify-center flex-shrink-0">
          <Medal className="w-6 h-6 text-nature" />
        </div>
        <div className="flex-1">
          <h3 className={`text-sm font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Weekly Challenge
          </h3>
          <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            {description}
          </p>
        </div>
        <span className="bg-nature text-white text-xs font-bold px-2 py-1 rounded-full">
          {current}/{total}
        </span>
      </div>
      <div className="w-full bg-gray-300 dark:bg-gray-700 rounded-full h-2">
        <div 
          className="bg-nature h-2 rounded-full transition-all duration-300"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
