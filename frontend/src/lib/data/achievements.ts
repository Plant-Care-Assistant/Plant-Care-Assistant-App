import type { AchievementRequirement, GamificationState } from '@/lib/gamification/types';
import { levelFromXp } from '@/lib/gamification/level';

export type AchievementCategory =
  | 'collection'
  | 'scanning'
  | 'care'
  | 'streak'
  | 'level'
  | 'variety'
  | 'health'
  | 'special';

export interface Achievement {
  id: string;
  title: string;
  hint: string;
  iconSrc: string;
  category: AchievementCategory;
  requirement: AchievementRequirement;
}

export const ACHIEVEMENT_CATEGORY_LABELS: Record<AchievementCategory, string> = {
  collection: 'Collection',
  scanning: 'Scanning',
  care: 'Care & Watering',
  streak: 'Streaks',
  level: 'Level & XP',
  variety: 'Variety',
  health: 'Plant Health',
  special: 'Special',
};

export const ACHIEVEMENTS: Achievement[] = [
  {
    id: 'first-sprout',
    title: 'First Sprout',
    hint: 'Add your first plant to the collection',
    iconSrc: '/ic-6.png',
    category: 'collection',
    requirement: { type: 'counter', counter: 'plantsAdded', gte: 1 },
  },
  {
    id: 'small-garden',
    title: 'Small Garden',
    hint: 'Own 5 plants at once',
    iconSrc: '/ic-6.png',
    category: 'collection',
    requirement: { type: 'counter', counter: 'plantsAdded', gte: 5 },
  },
  {
    id: 'green-house',
    title: 'Green House',
    hint: 'Grow your collection to 10 plants',
    iconSrc: '/ic-6.png',
    category: 'collection',
    requirement: { type: 'counter', counter: 'plantsAdded', gte: 10 },
  },
  {
    id: 'jungle-keeper',
    title: 'Jungle Keeper',
    hint: 'Own 20 plants at once',
    iconSrc: '/ic-6.png',
    category: 'collection',
    requirement: { type: 'counter', counter: 'plantsAdded', gte: 20 },
  },

  {
    id: 'curious-eye',
    title: 'Curious Eye',
    hint: 'Scan your first plant',
    iconSrc: '/ic-camera.png',
    category: 'scanning',
    requirement: { type: 'counter', counter: 'plantsScanned', gte: 1 },
  },
  {
    id: 'plant-detective',
    title: 'Plant Detective',
    hint: 'Scan 10 plants in total',
    iconSrc: '/ic-camera.png',
    category: 'scanning',
    requirement: { type: 'counter', counter: 'plantsScanned', gte: 10 },
  },
  {
    id: 'field-researcher',
    title: 'Field Researcher',
    hint: 'Scan 25 plants without adding them to your collection',
    iconSrc: '/ic-camera.png',
    category: 'scanning',
    requirement: { type: 'counter', counter: 'plantsScannedNotAdded', gte: 25 },
  },
  {
    id: 'encyclopedia',
    title: 'Encyclopedia',
    hint: 'Identify 50 different species through scanning',
    iconSrc: '/ic-camera.png',
    category: 'scanning',
    requirement: { type: 'counter', counter: 'speciesScanned', gte: 50 },
  },

  {
    id: 'first-drop',
    title: 'First Drop',
    hint: 'Water a plant for the first time',
    iconSrc: '/ic-hand.png',
    category: 'care',
    requirement: { type: 'counter', counter: 'plantsWatered', gte: 1 },
  },
  {
    id: 'hydration-hero',
    title: 'Hydration Hero',
    hint: 'Water your plants 25 times',
    iconSrc: '/ic-hand.png',
    category: 'care',
    requirement: { type: 'counter', counter: 'plantsWatered', gte: 25 },
  },
  {
    id: 'perfect-schedule',
    title: 'Perfect Schedule',
    hint: 'Complete 50 care tasks on time',
    iconSrc: '/ic-hand.png',
    category: 'care',
    requirement: { type: 'counter', counter: 'careTasksCompleted', gte: 50 },
  },

  {
    id: 'getting-started',
    title: 'Getting Started',
    hint: 'Maintain a 3-day care streak',
    iconSrc: '/ic-7.png',
    category: 'streak',
    requirement: { type: 'counter', counter: 'currentStreak', gte: 3 },
  },
  {
    id: 'week-warrior',
    title: 'Week Warrior',
    hint: 'Maintain a 7-day care streak',
    iconSrc: '/ic-7.png',
    category: 'streak',
    requirement: { type: 'counter', counter: 'currentStreak', gte: 7 },
  },
  {
    id: 'monthly-master',
    title: 'Monthly Master',
    hint: 'Keep a 30-day streak going',
    iconSrc: '/ic-7.png',
    category: 'streak',
    requirement: { type: 'counter', counter: 'currentStreak', gte: 30 },
  },
  {
    id: 'centurion',
    title: 'Centurion',
    hint: 'Reach a 100-day care streak',
    iconSrc: '/ic-7.png',
    category: 'streak',
    requirement: { type: 'counter', counter: 'longestStreak', gte: 100 },
  },

  {
    id: 'rookie-gardener',
    title: 'Rookie Gardener',
    hint: 'Reach level 5',
    iconSrc: '/ic-3.png',
    category: 'level',
    requirement: { type: 'level', gte: 5 },
  },
  {
    id: 'plant-expert',
    title: 'Plant Expert',
    hint: 'Reach level 10',
    iconSrc: '/ic-3.png',
    category: 'level',
    requirement: { type: 'level', gte: 10 },
  },
  {
    id: 'botanical-guru',
    title: 'Botanical Guru',
    hint: 'Reach level 25',
    iconSrc: '/ic-3.png',
    category: 'level',
    requirement: { type: 'level', gte: 25 },
  },

  {
    id: 'variety-pack',
    title: 'Variety Pack',
    hint: 'Add 5 plants to your collection',
    iconSrc: '/ic-1.png',
    category: 'variety',
    requirement: { type: 'counter', counter: 'speciesOwned', gte: 5 },
  },
  {
    id: 'diverse-garden',
    title: 'Diverse Garden',
    hint: 'Add 10 plants to your collection',
    iconSrc: '/ic-1.png',
    category: 'variety',
    requirement: { type: 'counter', counter: 'speciesOwned', gte: 10 },
  },

  {
    id: 'green-thumb',
    title: 'Green Thumb',
    hint: 'Keep 5 plants healthy for a full week',
    iconSrc: '/ic-2.png',
    category: 'health',
    requirement: { type: 'counter', counter: 'plantsWatered', gte: 35 },
  },
  {
    id: 'plant-whisperer',
    title: 'Plant Whisperer',
    hint: 'All plants 100% healthy for 30 days',
    iconSrc: '/ic-2.png',
    category: 'health',
    requirement: { type: 'counter', counter: 'careTasksCompleted', gte: 150 },
  },

  {
    id: 'welcome-aboard',
    title: 'Welcome Aboard',
    hint: 'Create your Plant Care account',
    iconSrc: '/ic-4.png',
    category: 'special',
    requirement: { type: 'flag', flag: 'accountCreated' },
  },
  {
    id: 'profile-pro',
    title: 'Profile Pro',
    hint: 'Complete every field on your profile',
    iconSrc: '/ic-5.png',
    category: 'special',
    requirement: { type: 'flag', flag: 'profileComplete' },
  },
  {
    id: 'early-bird',
    title: 'Early Bird',
    hint: 'Water a plant before 9 AM',
    iconSrc: '/ic-8.png',
    category: 'special',
    requirement: { type: 'counter', counter: 'watersBefore9AM', gte: 1 },
  },
];

export const ACHIEVEMENT_CATEGORY_ORDER: AchievementCategory[] = [
  'collection',
  'scanning',
  'care',
  'streak',
  'level',
  'variety',
  'health',
  'special',
];

export function isAchievementUnlocked(achievement: Achievement, state: GamificationState): boolean {
  const { requirement } = achievement;
  if (requirement.type === 'counter') {
    return state.counters[requirement.counter] >= requirement.gte;
  }
  if (requirement.type === 'level') {
    return levelFromXp(state.xp) >= requirement.gte;
  }
  return state.flags[requirement.flag] === true;
}
