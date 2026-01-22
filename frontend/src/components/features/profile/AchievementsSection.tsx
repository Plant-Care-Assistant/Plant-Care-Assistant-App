import { AchievementCard } from './AchievementCard';

interface AchievementsSectionProps {
  darkMode?: boolean;
}

export function AchievementsSection({ darkMode = false }: AchievementsSectionProps) {
  return (
    <div className={`rounded-3xl shadow-lg p-6 sm:p-8 ${darkMode ? 'bg-neutral-800' : 'bg-white'}`}>
      <div className="flex justify-between items-center mb-6 sm:mb-8">
        <h2 className={`text-xl sm:text-2xl font-bold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
          Achievements
        </h2>
        <span className={`text-sm sm:text-base font-semibold ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
          3/6
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4">
        <AchievementCard
          imageSrc="/ic-6.png"
          title="First Plant"
          description="Add your first plant"
          unlocked={true}
          darkMode={darkMode}
        />
        <AchievementCard
          imageSrc="/ic-7.png"
          title="Week Streak"
          description="7 day care streak"
          unlocked={true}
          darkMode={darkMode}
        />
        <AchievementCard
          imageSrc="/ic-3.png"
          title="Plant Expert"
          description="Reach level 10"
          unlocked={true}
          darkMode={darkMode}
        />
        <AchievementCard
          title="Green Thumb"
          description="Keep 5 plants healthy"
          unlocked={false}
          darkMode={darkMode}
        />
        <AchievementCard
          title="Botanist"
          description="Identify 50 plants"
          unlocked={false}
          darkMode={darkMode}
        />
        <AchievementCard
          title="Collector"
          description="Own 20 plants"
          unlocked={false}
          darkMode={darkMode}
        />
      </div>
    </div>
  );
}
