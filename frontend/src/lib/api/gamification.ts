import { apiClient } from "./client";
import type { XpActionId } from "@/lib/gamification/xp-actions";
import type { CounterName, FlagName } from "@/lib/gamification/types";

/**
 * Snake_case backend counter name → camelCase frontend name.
 * Keys match `services/gamification.py:COUNTERS`.
 */
const COUNTER_MAP: Record<string, CounterName> = {
  plants_added: "plantsAdded",
  plants_scanned: "plantsScanned",
  plants_scanned_not_added: "plantsScannedNotAdded",
  plants_watered: "plantsWatered",
  care_tasks_completed: "careTasksCompleted",
  species_owned: "speciesOwned",
  species_scanned: "speciesScanned",
  waters_before_9am: "watersBefore9AM",
};

/**
 * Backend stores flags as the Python enum `name` (lowercase), e.g. `first_login`.
 * Frontend uses semantic names like `accountCreated`. Mapping bridges the two.
 */
const FLAG_MAP: Record<string, FlagName> = {
  first_login: "accountCreated",
  complete_profile: "profileComplete",
  first_home_visit: "firstHomeVisit",
  first_collection_visit: "firstCollectionVisit",
  first_scan_visit: "firstScanVisit",
  first_profile_visit: "firstProfileVisit",
  first_theme_change: "firstThemeChange",
};

/** Raw shape returned by `GET /gamification/me` and inside `POST /events` response. */
interface BackendSnapshot {
  xp: number;
  level: number;
  counters: Record<string, number>;
  flags: Record<string, boolean>;
  unlocked_achievement_ids: string[];
  current_streak: number;
  longest_streak: number;
  last_active_date: string | null;
}

interface BackendEventResponse {
  snapshot: BackendSnapshot;
  xp_awarded: number;
  newly_unlocked: string[];
}

export interface NormalizedSnapshot {
  xp: number;
  level: number;
  counters: Record<CounterName, number>;
  flags: Record<FlagName, boolean>;
  unlockedAchievementIds: string[];
  currentStreak: number;
  longestStreak: number;
  lastActiveDate: string | null;
}

export interface NormalizedEventResponse {
  snapshot: NormalizedSnapshot;
  xpAwarded: number;
  newlyUnlocked: string[];
}

const EMPTY_COUNTERS: Record<CounterName, number> = {
  plantsAdded: 0,
  plantsScanned: 0,
  plantsScannedNotAdded: 0,
  plantsWatered: 0,
  careTasksCompleted: 0,
  speciesOwned: 0,
  speciesScanned: 0,
  currentStreak: 0,
  longestStreak: 0,
  watersBefore9AM: 0,
};

const EMPTY_FLAGS: Record<FlagName, boolean> = {
  accountCreated: false,
  profileComplete: false,
  firstHomeVisit: false,
  firstCollectionVisit: false,
  firstScanVisit: false,
  firstProfileVisit: false,
  firstThemeChange: false,
};

function normalizeSnapshot(snap: BackendSnapshot): NormalizedSnapshot {
  const counters: Record<CounterName, number> = { ...EMPTY_COUNTERS };
  for (const [key, val] of Object.entries(snap.counters)) {
    const feKey = COUNTER_MAP[key];
    if (feKey) counters[feKey] = val;
  }
  // Streaks live on the snapshot root, not in `counters`, but the frontend
  // GamificationState surfaces them through counters for legacy reasons.
  counters.currentStreak = snap.current_streak;
  counters.longestStreak = snap.longest_streak;

  const flags: Record<FlagName, boolean> = { ...EMPTY_FLAGS };
  for (const [key, val] of Object.entries(snap.flags)) {
    const feKey = FLAG_MAP[key];
    if (feKey) flags[feKey] = val;
  }

  return {
    xp: snap.xp,
    level: snap.level,
    counters,
    flags,
    unlockedAchievementIds: snap.unlocked_achievement_ids,
    currentStreak: snap.current_streak,
    longestStreak: snap.longest_streak,
    lastActiveDate: snap.last_active_date,
  };
}

export const gamificationApi = {
  async getMe(): Promise<NormalizedSnapshot | null> {
    try {
      const res = await apiClient.get<BackendSnapshot>("/gamification/me");
      return normalizeSnapshot(res.data);
    } catch (err) {
      console.warn("gamificationApi.getMe failed:", err);
      return null;
    }
  },

  async postEvent(actionId: XpActionId): Promise<NormalizedEventResponse | null> {
    try {
      const res = await apiClient.post<BackendEventResponse>("/gamification/events", {
        action_id: actionId,
        client_tz_offset_min: -new Date().getTimezoneOffset(),
      });
      return {
        snapshot: normalizeSnapshot(res.data.snapshot),
        xpAwarded: res.data.xp_awarded,
        newlyUnlocked: res.data.newly_unlocked,
      };
    } catch (err) {
      console.warn(`gamificationApi.postEvent(${actionId}) failed:`, err);
      return null;
    }
  },
};
