'use client';

import { useState } from 'react';
import { useAuth, useGamification, useTheme } from '@/providers';
import { ProfileHeader } from '@/components/features/profile/ProfileHeader';
import { StatsGrid } from '@/components/features/profile/StatsGrid';
import { AchievementsSection } from '@/components/features/profile/AchievementsSection';
import { AchievementProgress } from '@/components/features/profile/AchievementProgress';
import { SettingsSection } from '@/components/features/profile/SettingsSection';
import { ACHIEVEMENTS } from '@/lib/data/achievements';

const TOTAL_ACHIEVEMENTS = ACHIEVEMENTS.length;

export interface ProfileScreenProps {
  onDarkModeToggle?: (enabled: boolean) => void;
}

export function ProfileScreen({ onDarkModeToggle }: ProfileScreenProps) {
  const { user } = useAuth();
  const { theme } = useTheme();
  const darkMode = theme === 'dark';
  const { state, level, unlockedCount } = useGamification();
  const [careRemindersEnabled, setCareRemindersEnabled] = useState(true);
  const [weatherTipsEnabled, setWeatherTipsEnabled] = useState(true);

  const streak = state.counters.currentStreak;

  return (
    <div className="min-h-screen pb-24 lg:pb-8">
      <div className="p-4 lg:p-6 max-w-7xl mx-auto">
        <div className="space-y-6">
          <ProfileHeader
            name={user?.username || 'User'}
            level={level}
            totalXP={state.xp}
            dayStreak={streak}
            achievements={`${unlockedCount}/${TOTAL_ACHIEVEMENTS}`}
            darkMode={darkMode}
          />

          <StatsGrid
            level={level}
            dayStreak={streak}
            totalXP={state.xp}
            darkMode={darkMode}
          />

          <AchievementsSection darkMode={darkMode} />

          <AchievementProgress
            unlocked={unlockedCount}
            total={TOTAL_ACHIEVEMENTS}
            darkMode={darkMode}
          />

          <SettingsSection
            darkMode={darkMode}
            onDarkModeToggle={onDarkModeToggle}
            careRemindersEnabled={careRemindersEnabled}
            onCareRemindersToggle={setCareRemindersEnabled}
            weatherTipsEnabled={weatherTipsEnabled}
            onWeatherTipsToggle={setWeatherTipsEnabled}
          />
        </div>
      </div>
    </div>
  );
}
