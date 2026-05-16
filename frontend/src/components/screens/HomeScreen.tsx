'use client';

import { motion } from 'framer-motion';
import { LevelBadge } from '@/components/features/home/LevelBadge';
import { StreakCard } from '@/components/features/home/StreakCard';
import { HealthyPlantsCard } from '@/components/features/home/HealthyPlantsCard';
import { XPEarnedCard } from '@/components/features/home/XPEarnedCard';
import { HeaderGreeting } from '@/components/features/home/HeaderGreeting';
import { HeaderIcon } from '@/components/features/home/HeaderIcon';
import { WeeklyChallenge } from '@/components/features/home/WeeklyChallenge';
import { WeatherTip } from '@/components/features/home/WeatherTip';
import { AttentionSummaryCards } from '@/components/features/home/AttentionSummaryCards';
import { NeedsCareList } from '@/components/features/home/NeedsCareList';
import { UserPlant } from '@/types';
import { useGamification, useTheme } from '@/providers';
import { useWeeklyChallenge } from '@/lib/gamification/useWeeklyChallenge';

export interface HomeScreenProps {
  plants: UserPlant[];
}

export function HomeScreen({ plants }: HomeScreenProps) {
  const { theme } = useTheme();
  const darkMode = theme === 'dark';
  const totalPlants = plants.length;

  const plantsNeedingWater = plants.filter(
    (p) => p.days_until_water != null && p.days_until_water <= 0,
  );
  const plantsNeedWater = plantsNeedingWater.length;
  const diseasedCount = plants.filter((p) => p.last_health_label === 'diseased').length;
  // null = no plants yet; card shows "—" instead of a misleading percentage.
  const healthyPlantsCount: number | null =
    totalPlants > 0 ? Math.round(((totalPlants - diseasedCount) / totalPlants) * 100) : null;

  const { state, level } = useGamification();
  const streak = state.counters.currentStreak;
  const totalXp = state.xp;
  const challenge = useWeeklyChallenge(streak);

  return (
    <div className={`p-4 lg:p-6 pb-24 lg:pb-4 max-w-7xl mx-auto ${darkMode ? 'text-white' : 'text-gray-900'}`}>
      <motion.div
        className="mb-6 mt-1"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center justify-between mb-4 h-20">
          <HeaderGreeting darkMode={darkMode} />
          <HeaderIcon level={level} darkMode={darkMode} />
        </div>
      </motion.div>

      <div className="mb-4">
        <LevelBadge level={level} darkMode={darkMode} />
      </div>

      <div className="grid grid-cols-3 gap-3 my-6">
        <StreakCard streak={streak} darkMode={darkMode} />
        <HealthyPlantsCard percentage={healthyPlantsCount} darkMode={darkMode} />
        <XPEarnedCard xp={totalXp} darkMode={darkMode} />
      </div>

      <WeeklyChallenge
        current={challenge.current}
        total={challenge.total}
        description={challenge.description}
        darkMode={darkMode}
      />

      <WeatherTip
        title="Weather Tip" 
        description="Cloudy today! Your plants might need less water. Check the soil first." 
        darkMode={darkMode} 
      />

      <div className="mb-6">
        <h2 className={`text-lg font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
          Needs care today
        </h2>

        {totalPlants === 0 ? (
          <p className={`text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
            No plants yet. Add some to your collection!
          </p>
        ) : (
          <div className="mb-4">
            <NeedsCareList plants={plantsNeedingWater} darkMode={darkMode} />
          </div>
        )}

        {totalPlants > 0 && (
          <AttentionSummaryCards
            needWater={plantsNeedWater}
            diseased={diseasedCount}
            darkMode={darkMode}
          />
        )}
      </div>
    </div>
  );
}
