'use client';

import { AchievementCard } from './AchievementCard';
import {
  ACHIEVEMENTS,
  ACHIEVEMENT_CATEGORY_LABELS,
  ACHIEVEMENT_CATEGORY_ORDER,
  type Achievement,
  type AchievementCategory,
} from '@/lib/data/achievements';
import { useGamification } from '@/providers';

interface AchievementsSectionProps {
  darkMode?: boolean;
}

const ACHIEVEMENTS_BY_CATEGORY: ReadonlyArray<{
  category: AchievementCategory;
  items: Achievement[];
}> = ACHIEVEMENT_CATEGORY_ORDER
  .map((category) => ({ category, items: ACHIEVEMENTS.filter((a) => a.category === category) }))
  .filter((group) => group.items.length > 0);

const TOTAL_ACHIEVEMENTS = ACHIEVEMENTS.length;

export function AchievementsSection({ darkMode = false }: AchievementsSectionProps) {
  const { isAchievementUnlocked, unlockedCount } = useGamification();

  return (
    <div className={`rounded-3xl shadow-lg p-6 sm:p-8 ${darkMode ? 'bg-neutral-800' : 'bg-white'}`}>
      <div className="flex justify-between items-center mb-6 sm:mb-8">
        <h2 className={`text-xl sm:text-2xl font-bold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
          Achievements
        </h2>
        <span className={`text-sm sm:text-base font-semibold ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
          {unlockedCount}/{TOTAL_ACHIEVEMENTS}
        </span>
      </div>

      <div className="space-y-6 sm:space-y-8">
        {ACHIEVEMENTS_BY_CATEGORY.map(({ category, items }) => (
          <div key={category}>
            <h3 className={`text-sm sm:text-base font-semibold uppercase tracking-wide mb-3 sm:mb-4 ${
              darkMode ? 'text-neutral-400' : 'text-neutral-500'
            }`}>
              {ACHIEVEMENT_CATEGORY_LABELS[category]}
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4">
              {items.map((achievement) => (
                <AchievementCard
                  key={achievement.id}
                  imageSrc={achievement.iconSrc}
                  title={achievement.title}
                  description={achievement.hint}
                  unlocked={isAchievementUnlocked(achievement.id)}
                  darkMode={darkMode}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
