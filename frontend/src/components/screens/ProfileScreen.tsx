'use client';

import { useEffect, useState } from 'react';
import { useAuth, useGamification, useTheme } from '@/providers';
import { ProfileHeader } from '@/components/features/profile/ProfileHeader';
import { StatsGrid } from '@/components/features/profile/StatsGrid';
import { AchievementsSection } from '@/components/features/profile/AchievementsSection';
import { AchievementProgress } from '@/components/features/profile/AchievementProgress';
import { SettingsSection } from '@/components/features/profile/SettingsSection';
import { ACHIEVEMENTS } from '@/lib/data/achievements';
import {
  loadUserSettings,
  saveUserSettings,
  notificationsSupported,
} from '@/lib/notifications/careReminders';

const TOTAL_ACHIEVEMENTS = ACHIEVEMENTS.length;

export interface ProfileScreenProps {
  onDarkModeToggle?: (enabled: boolean) => void;
}

export function ProfileScreen({ onDarkModeToggle }: ProfileScreenProps) {
  const { user } = useAuth();
  const { theme } = useTheme();
  const darkMode = theme === 'dark';
  const { state, level, xpIntoLevel, xpForNext, unlockedCount } = useGamification();
  // Initialise from localStorage cache so the toggle shows the right state
  // immediately, before the async backend fetch completes (fixes flash of wrong state).
  const [careRemindersEnabled, setCareRemindersEnabled] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    return localStorage.getItem('care_reminders_enabled') === 'true';
  });
  const [weatherTipsEnabled, setWeatherTipsEnabled] = useState(true);

  const streak = state.counters.currentStreak;

  const [notifHint, setNotifHint] = useState<string>(() => {
    if (typeof window === 'undefined') return 'Daily notifications';
    if ('Notification' in window && Notification.permission === 'denied') {
      return 'Blocked — enable in browser site settings';
    }
    return 'Daily notifications';
  });

  // Load authoritative values from backend and sync both toggles.
  useEffect(() => {
    loadUserSettings().then((s) => {
      setCareRemindersEnabled(s.care_reminders);
      setWeatherTipsEnabled(s.weather_tips);
      if (notificationsSupported() && Notification.permission === 'denied' && s.care_reminders) {
        setNotifHint('Blocked — enable in browser site settings');
      }
    });
  }, []);

  // Non-async so the requestPermission() call stays as close to the click as possible.
  const handleCareRemindersToggle = (value: boolean) => {
    if (!value) {
      setCareRemindersEnabled(false);
      saveUserSettings({ care_reminders: false, weather_tips: weatherTipsEnabled });
      setNotifHint('Daily notifications');
      return;
    }

    if (!notificationsSupported()) {
      setNotifHint('Not supported by your browser');
      return;
    }

    // Let the browser decide whether to show the dialog based on current permission.
    // Calling this directly in the sync handler satisfies the user-gesture requirement.
    Notification.requestPermission().then((result) => {
      if (result === 'granted') {
        setCareRemindersEnabled(true);
        saveUserSettings({ care_reminders: true, weather_tips: weatherTipsEnabled });
        setNotifHint('Daily notifications');

        // Welcome notification — confirms to the user that it worked.
        new Notification('🌿 Plant Care', {
          body: "Care reminders are on! I'll let you know when your plants need water. 💧",
          icon: '/logo.png',
          tag: 'welcome-reminder',
        });
      } else {
        setNotifHint('Blocked — enable in browser site settings');
      }
    });
  };

  return (
    <div className="min-h-screen pb-24 lg:pb-8">
      <div className="p-4 lg:p-6 max-w-7xl mx-auto">
        <div className="space-y-6">
          <ProfileHeader
            name={user?.username || 'User'}
            level={level}
            xpIntoLevel={xpIntoLevel}
            xpForNext={xpForNext}
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
            onCareRemindersToggle={handleCareRemindersToggle}
            careRemindersHint={notifHint}
            weatherTipsEnabled={weatherTipsEnabled}
            onWeatherTipsToggle={(v) => {
              setWeatherTipsEnabled(v);
              saveUserSettings({ care_reminders: careRemindersEnabled, weather_tips: v });
            }}
          />
        </div>
      </div>
    </div>
  );
}
