import type { CounterName, FlagName } from './types';

export type XpActionId =
  | 'SCAN_IDENTIFY'
  | 'SCAN_AND_ADD'
  | 'ADD_PLANT'
  | 'WATER_PLANT'
  | 'COMPLETE_CARE_TASK'
  | 'WATER_BEFORE_9AM'
  | 'FIRST_LOGIN'
  | 'COMPLETE_PROFILE'
  | 'FIRST_HOME_VISIT'
  | 'FIRST_COLLECTION_VISIT'
  | 'FIRST_SCAN_VISIT'
  | 'FIRST_PROFILE_VISIT'
  | 'FIRST_THEME_CHANGE'
  | 'DAILY_LOGIN_BONUS'
  | 'ACHIEVEMENT_UNLOCK';

export interface XpAction {
  id: XpActionId;
  xp: number;
  label: string;
  description: string;
  counters?: CounterName[];
  flag?: FlagName;
  onceOnly?: boolean;
}

export const XP_ACTIONS: Record<XpActionId, XpAction> = {
  SCAN_IDENTIFY: {
    id: 'SCAN_IDENTIFY',
    xp: 25,
    label: 'Plant Identified',
    description: 'Scanned a plant without adding it to collection',
    counters: ['plantsScanned', 'plantsScannedNotAdded', 'speciesScanned'],
  },
  SCAN_AND_ADD: {
    id: 'SCAN_AND_ADD',
    xp: 100,
    label: 'Scan + Collected',
    description: 'Identified and added a plant in one flow',
    counters: ['plantsScanned', 'plantsAdded', 'speciesScanned', 'speciesOwned'],
  },
  ADD_PLANT: {
    id: 'ADD_PLANT',
    xp: 75,
    label: 'Plant Added',
    description: 'Added a plant to your collection',
    counters: ['plantsAdded', 'speciesOwned'],
  },
  WATER_PLANT: {
    id: 'WATER_PLANT',
    xp: 15,
    label: 'Plant Watered',
    description: 'Watered one of your plants',
    counters: ['plantsWatered'],
  },
  COMPLETE_CARE_TASK: {
    id: 'COMPLETE_CARE_TASK',
    xp: 25,
    label: 'Care Task Done',
    description: 'Completed a scheduled care task',
    counters: ['careTasksCompleted'],
  },
  WATER_BEFORE_9AM: {
    id: 'WATER_BEFORE_9AM',
    xp: 10,
    label: 'Early Bird Bonus',
    description: 'Bonus for watering before 9 AM',
    counters: ['watersBefore9AM'],
  },
  FIRST_LOGIN: {
    id: 'FIRST_LOGIN',
    xp: 50,
    label: 'Welcome!',
    description: 'Joined Plant Care',
    flag: 'accountCreated',
    onceOnly: true,
  },
  COMPLETE_PROFILE: {
    id: 'COMPLETE_PROFILE',
    xp: 30,
    label: 'Profile Complete',
    description: 'Filled in every profile field',
    flag: 'profileComplete',
    onceOnly: true,
  },
  FIRST_HOME_VISIT: {
    id: 'FIRST_HOME_VISIT',
    xp: 10,
    label: 'Explorer',
    description: 'Visited the home screen',
    flag: 'firstHomeVisit',
    onceOnly: true,
  },
  FIRST_COLLECTION_VISIT: {
    id: 'FIRST_COLLECTION_VISIT',
    xp: 10,
    label: 'Explorer',
    description: 'Opened your plant collection',
    flag: 'firstCollectionVisit',
    onceOnly: true,
  },
  FIRST_SCAN_VISIT: {
    id: 'FIRST_SCAN_VISIT',
    xp: 10,
    label: 'Explorer',
    description: 'Tried the scan screen',
    flag: 'firstScanVisit',
    onceOnly: true,
  },
  FIRST_PROFILE_VISIT: {
    id: 'FIRST_PROFILE_VISIT',
    xp: 10,
    label: 'Explorer',
    description: 'Checked out your profile',
    flag: 'firstProfileVisit',
    onceOnly: true,
  },
  FIRST_THEME_CHANGE: {
    id: 'FIRST_THEME_CHANGE',
    xp: 15,
    label: 'Personalized',
    description: 'Changed the app theme',
    flag: 'firstThemeChange',
    onceOnly: true,
  },
  DAILY_LOGIN_BONUS: {
    id: 'DAILY_LOGIN_BONUS',
    xp: 20,
    label: 'Daily Bonus',
    description: 'Back again today — streak +1',
  },
  ACHIEVEMENT_UNLOCK: {
    id: 'ACHIEVEMENT_UNLOCK',
    xp: 50,
    label: 'Achievement Unlocked',
    description: 'Bonus XP for unlocking an achievement',
  },
};
