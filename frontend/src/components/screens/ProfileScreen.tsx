'use client';

import { useState } from 'react';
import { ProfileHeader } from '@/components/features/profile/ProfileHeader';
import { StatsGrid } from '@/components/features/profile/StatsGrid';
import { AchievementsSection } from '@/components/features/profile/AchievementsSection';
import { AchievementProgress } from '@/components/features/profile/AchievementProgress';
import { SettingsSection } from '@/components/features/profile/SettingsSection';

export interface ProfileScreenProps {
  darkMode?: boolean;
  onDarkModeToggle?: (enabled: boolean) => void;
}

export function ProfileScreen({ darkMode = false, onDarkModeToggle }: ProfileScreenProps) {
  const [careRemindersEnabled, setCareRemindersEnabled] = useState(true);
  const [weatherTipsEnabled, setWeatherTipsEnabled] = useState(true);

  const userData = {
    name: 'Sarah',
    level: 12,
    totalXP: 2450,
    dayStreak: 7,
    achievements: '3/6',
  };

  return (
    <div className="min-h-screen pb-24 lg:pb-8">
      <div className="p-4 lg:p-6 max-w-7xl mx-auto">
        <div className="space-y-6">
          {/* Profile Header Card */}
          <ProfileHeader
            name={userData.name}
            level={userData.level}
            totalXP={userData.totalXP}
            dayStreak={userData.dayStreak}
            achievements={userData.achievements}
            darkMode={darkMode}
          />

          {/* Stats Cards Grid */}
          <StatsGrid
            level={userData.level}
            dayStreak={userData.dayStreak}
            totalXP={userData.totalXP}
            darkMode={darkMode}
          />

          {/* Achievements Section */}
          <AchievementsSection darkMode={darkMode} />

          {/* Achievement Progress Section */}
          <AchievementProgress
            unlocked={3}
            total={6}
            darkMode={darkMode}
          />

          {/* Settings Section */}
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
