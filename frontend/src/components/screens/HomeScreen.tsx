'use client';

import { motion } from 'framer-motion';
import { useMemo } from 'react';
import { LevelBadge } from '@/components/features/home/LevelBadge';
import { DailyGoal } from '@/components/features/home/DailyGoal';
import { StreakCard } from '@/components/features/home/StreakCard';
import { HealthyPlantsCard } from '@/components/features/home/HealthyPlantsCard';
import { XPEarnedCard } from '@/components/features/home/XPEarnedCard';
import { HeaderGreeting } from '@/components/features/home/HeaderGreeting';
import { HeaderIcon } from '@/components/features/home/HeaderIcon';
import { WeeklyChallenge } from '@/components/features/home/WeeklyChallenge';
import { WeatherTip } from '@/components/features/home/WeatherTip';
import { PlantAttentionCard } from '@/components/features/home/PlantAttentionCard';
import { AttentionSummaryCards } from '@/components/features/home/AttentionSummaryCards';
import { UserPlant } from '@/types';

export interface HomeScreenProps {
  darkMode: boolean;
  plants: UserPlant[];
}

export function HomeScreen({ darkMode, plants }: HomeScreenProps) {
  // With real data we don't have health status yet - show plant count based stats
  const totalPlants = plants.length;
  const healthyPlantsCount = totalPlants > 0 ? 100 : 0;
  const plantsNeedWater = 0;

  return (
    <div className={`p-4 lg:p-6 pb-24 lg:pb-4 max-w-7xl mx-auto ${darkMode ? 'text-white' : 'text-gray-900'}`}>
      {/* Header */}
      <motion.div 
        className="mb-6 mt-1"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center justify-between mb-4 h-20">
          <HeaderGreeting darkMode={darkMode} />
          <HeaderIcon darkMode={darkMode} />
        </div>
      </motion.div>

      {/* Level Badge */}
      <div className="mb-4">
        <LevelBadge level={12} darkMode={darkMode} />
      </div>

      {/* Daily Goal Widget */}
      <DailyGoal current={2450} total={3000} darkMode={darkMode} />

      {/* Stats Grid */}
      <div className="grid grid-cols-3 gap-3 my-6">
        <StreakCard streak={7} darkMode={darkMode} />
        <HealthyPlantsCard percentage={healthyPlantsCount} darkMode={darkMode} />
        <XPEarnedCard xp={2450} darkMode={darkMode} />
      </div>

      {/* Weekly Challenge */}
      <WeeklyChallenge 
        current={4} 
        total={7} 
        description="Water 7 plants this week" 
        darkMode={darkMode} 
      />

      {/* Weather Tip */}
      <WeatherTip 
        title="Weather Tip" 
        description="Cloudy today! Your plants might need less water. Check the soil first." 
        darkMode={darkMode} 
      />

      {/* Need Attention Section */}
      <div className="mb-6">
        <h2 className={`text-lg font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
          Need Attention
        </h2>
        
        {/* Plant cards needing attention */}
        <div className="mb-4">
          {totalPlants === 0 && (
            <p className={`text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
              No plants yet. Add some to your collection!
            </p>
          )}
        </div>

        {/* Attention Summary Cards */}
        <AttentionSummaryCards 
          needWater={plantsNeedWater} 
          lowLight={2} 
          darkMode={darkMode} 
        />
      </div>
    </div>
  );
}
