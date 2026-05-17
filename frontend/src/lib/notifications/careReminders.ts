import { apiClient } from '@/lib/api/client';
import type { UserPlant } from '@/types';

const LAST_NOTIFIED_KEY = 'care_reminder_last_date';
const PREF_CACHE_KEY = 'care_reminders_enabled';

// ---------- Backend sync ----------

export interface UserSettings {
  care_reminders: boolean;
  weather_tips: boolean;
}

export async function loadUserSettings(): Promise<UserSettings> {
  try {
    const res = await apiClient.get<UserSettings>('/users/me/settings');
    const settings = {
      care_reminders: res.data.care_reminders ?? false,
      weather_tips: res.data.weather_tips ?? true,
    };
    localStorage.setItem(PREF_CACHE_KEY, String(settings.care_reminders));
    return settings;
  } catch {
    return {
      care_reminders: localStorage.getItem(PREF_CACHE_KEY) === 'true',
      weather_tips: true,
    };
  }
}

export async function saveUserSettings(settings: UserSettings): Promise<void> {
  localStorage.setItem(PREF_CACHE_KEY, String(settings.care_reminders));
  try {
    await apiClient.put('/users/me/settings', settings);
  } catch {
    // Silently fail; UI state already reflects the change.
  }
}

// ---------- Browser permission ----------

export function notificationsSupported(): boolean {
  return typeof window !== 'undefined' && 'Notification' in window;
}

// ---------- Firing reminders ----------

export function maybeSendCareReminder(plantsNeedingWater: UserPlant[]): void {
  if (!notificationsSupported()) return;
  if (Notification.permission !== 'granted') return;
  if (localStorage.getItem(PREF_CACHE_KEY) !== 'true') return;
  if (plantsNeedingWater.length === 0) return;

  const today = new Date().toDateString();
  if (localStorage.getItem(LAST_NOTIFIED_KEY) === today) return;

  const names = plantsNeedingWater.map(p => p.custom_name || 'Unnamed plant');
  const first3 = names.slice(0, 3);
  const rest = names.length - 3;

  let body: string;
  if (names.length === 1) {
    body = `${first3[0]} needs water today! 💧`;
  } else if (rest > 0) {
    body = `${first3.join(', ')} and ${rest} more need water today. 💧`;
  } else {
    body = `${first3.join(', ')} need water today. 💧`;
  }

  new Notification('🌿 Plant Care Reminder', {
    body,
    icon: '/logo.png',
    tag: 'care-reminder', // deduplicate — only one notification at a time
  });

  localStorage.setItem(LAST_NOTIFIED_KEY, today);
}
