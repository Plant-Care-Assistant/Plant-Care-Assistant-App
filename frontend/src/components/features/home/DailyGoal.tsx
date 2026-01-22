interface DailyGoalProps {
  current: number;
  total: number;
  darkMode: boolean;
}

export function DailyGoal({ current, total, darkMode }: DailyGoalProps) {
  const percentage = Math.round((current / total) * 100);

  return (
    <div className={`p-5 rounded-3xl ${darkMode ? 'bg-neutral-800' : 'bg-white'} shadow-md mb-5`}>
      <div className="flex justify-between items-center mb-3">
        <h3 className={`text-sm font-semibold ${darkMode ? 'text-neutral-300' : 'text-neutral-600'}`}>
          Daily Goal
        </h3>
        <span className={`text-sm font-bold ${darkMode ? 'text-neutral-300' : 'text-neutral-600'}`}>
          {current}/{total} XP
        </span>
      </div>
      <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-3">
        <div 
          className="bg-nature h-3 rounded-full transition-all duration-300"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
