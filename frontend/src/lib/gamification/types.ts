export type CounterName =
  | 'plantsAdded'
  | 'plantsScanned'
  | 'plantsScannedNotAdded'
  | 'plantsWatered'
  | 'careTasksCompleted'
  | 'speciesOwned'
  | 'speciesScanned'
  | 'currentStreak'
  | 'longestStreak'
  | 'watersBefore9AM';

export type FlagName =
  | 'accountCreated'
  | 'profileComplete'
  | 'firstHomeVisit'
  | 'firstCollectionVisit'
  | 'firstScanVisit'
  | 'firstProfileVisit'
  | 'firstThemeChange';

export type AchievementRequirement =
  | { type: 'counter'; counter: CounterName; gte: number }
  | { type: 'flag'; flag: FlagName }
  | { type: 'level'; gte: number };

export interface GamificationState {
  xp: number;
  counters: Record<CounterName, number>;
  flags: Record<FlagName, boolean>;
  unlockedAchievementIds: string[];
  lastActiveDate: string | null;
}

export const EMPTY_STATE: GamificationState = {
  xp: 0,
  counters: {
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
  },
  flags: {
    accountCreated: false,
    profileComplete: false,
    firstHomeVisit: false,
    firstCollectionVisit: false,
    firstScanVisit: false,
    firstProfileVisit: false,
    firstThemeChange: false,
  },
  unlockedAchievementIds: [],
  lastActiveDate: null,
};
