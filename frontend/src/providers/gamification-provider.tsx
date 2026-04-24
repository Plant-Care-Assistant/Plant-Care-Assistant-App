'use client';

import {
  createContext,
  useContext,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  ReactNode,
} from 'react';
import { useAuth } from './auth-provider';
import { useToast } from './toast-provider';
import {
  EMPTY_STATE,
  type GamificationState,
} from '@/lib/gamification/types';
import { XP_ACTIONS, type XpActionId } from '@/lib/gamification/xp-actions';
import { progressWithinLevel } from '@/lib/gamification/level';
import { ACHIEVEMENTS, isAchievementUnlocked } from '@/lib/data/achievements';

const STORAGE_PREFIX = 'gamification:';

interface GamificationContextType {
  state: GamificationState;
  hydrated: boolean;
  level: number;
  xpIntoLevel: number;
  xpForNext: number;
  progressPercent: number;
  unlockedCount: number;
  awardXP: (actionId: XpActionId, opts?: { subtitle?: string }) => void;
  isAchievementUnlocked: (achievementId: string) => boolean;
}

const GamificationContext = createContext<GamificationContextType | undefined>(undefined);

function storageKey(userId: string | null) {
  return `${STORAGE_PREFIX}${userId ?? 'anonymous'}`;
}

function loadState(userId: string | null): GamificationState {
  if (typeof window === 'undefined') return EMPTY_STATE;
  try {
    const raw = window.localStorage.getItem(storageKey(userId));
    if (!raw) return EMPTY_STATE;
    const parsed = JSON.parse(raw) as Partial<GamificationState>;
    return {
      ...EMPTY_STATE,
      ...parsed,
      counters: { ...EMPTY_STATE.counters, ...(parsed.counters ?? {}) },
      flags: { ...EMPTY_STATE.flags, ...(parsed.flags ?? {}) },
      unlockedAchievementIds: parsed.unlockedAchievementIds ?? [],
    };
  } catch {
    return EMPTY_STATE;
  }
}

function saveState(userId: string | null, state: GamificationState) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(storageKey(userId), JSON.stringify(state));
  } catch {
    // quota/serialization errors: state is recoverable from memory until next write
  }
}

export function GamificationProvider({ children }: { children: ReactNode }) {
  const { user, isAuthenticated, isLoading: authLoading } = useAuth();
  const { showXpToast, showAchievementToast } = useToast();
  const [state, setState] = useState<GamificationState>(EMPTY_STATE);
  const [hydrated, setHydrated] = useState(false);
  const stateRef = useRef<GamificationState>(EMPTY_STATE);
  const userIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (authLoading) return;
    const uid = user?.id ?? null;
    userIdRef.current = uid;
    let loaded = loadState(uid);

    // Existing users (account older than 10 min) shouldn't be forced through the
    // "welcome aboard" toast flow on a new device — pre-seed the flag silently.
    if (user && !loaded.flags.accountCreated) {
      const accountAgeMs = Date.now() - new Date(user.created_at).getTime();
      if (accountAgeMs > 10 * 60 * 1000) {
        const unlocked = new Set(loaded.unlockedAchievementIds);
        unlocked.add('welcome-aboard');
        loaded = {
          ...loaded,
          flags: { ...loaded.flags, accountCreated: true },
          unlockedAchievementIds: Array.from(unlocked),
        };
      }
    }

    stateRef.current = loaded;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- hydrating React state from localStorage
    setState(loaded);
    setHydrated(true);
  }, [user, authLoading]);

  useEffect(() => {
    if (!hydrated) return;
    saveState(userIdRef.current, state);
  }, [state, hydrated]);

  const awardXP = useCallback<GamificationContextType['awardXP']>(
    (actionId, opts) => {
      const action = XP_ACTIONS[actionId];
      if (!action) return;

      const prev = stateRef.current;
      if (action.onceOnly && action.flag && prev.flags[action.flag]) return;

      const newCounters = { ...prev.counters };
      action.counters?.forEach((c) => {
        newCounters[c] = (newCounters[c] ?? 0) + 1;
      });
      const newFlags = { ...prev.flags };
      if (action.flag) newFlags[action.flag] = true;

      let next: GamificationState = {
        ...prev,
        xp: prev.xp + action.xp,
        counters: newCounters,
        flags: newFlags,
      };

      const prevIds = new Set(prev.unlockedAchievementIds);
      const newlyUnlocked: string[] = [];
      for (const achievement of ACHIEVEMENTS) {
        if (!prevIds.has(achievement.id) && isAchievementUnlocked(achievement, next)) {
          newlyUnlocked.push(achievement.id);
        }
      }
      const bonusXp = newlyUnlocked.length * XP_ACTIONS.ACHIEVEMENT_UNLOCK.xp;
      if (newlyUnlocked.length > 0) {
        next = {
          ...next,
          xp: next.xp + bonusXp,
          unlockedAchievementIds: [...next.unlockedAchievementIds, ...newlyUnlocked],
        };
      }

      stateRef.current = next;
      setState(next);

      if (action.xp > 0) {
        showXpToast({
          amount: action.xp,
          title: action.label,
          subtitle: opts?.subtitle ?? action.description,
        });
      }
      newlyUnlocked.forEach((id) => {
        const a = ACHIEVEMENTS.find((x) => x.id === id);
        if (a) showAchievementToast({ title: a.title, subtitle: a.hint });
      });
      if (bonusXp > 0) {
        showXpToast({
          amount: bonusXp,
          title: XP_ACTIONS.ACHIEVEMENT_UNLOCK.label,
          subtitle: `${newlyUnlocked.length} achievement${newlyUnlocked.length > 1 ? 's' : ''} unlocked`,
        });
      }
    },
    [showXpToast, showAchievementToast],
  );

  useEffect(() => {
    if (!isAuthenticated || !hydrated) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- one-shot welcome grant gated by flag
    if (!state.flags.accountCreated) awardXP('FIRST_LOGIN');
  }, [isAuthenticated, hydrated, state.flags.accountCreated, awardXP]);

  // Daily streak + login bonus: runs once per local day when the user is active.
  // Guarded by lastActiveDate so remounts within the same day don't re-award.
  useEffect(() => {
    if (!isAuthenticated || !hydrated) return;
    const today = new Date().toISOString().slice(0, 10);
    const prev = stateRef.current;
    if (prev.lastActiveDate === today) return;

    let newStreak = 1;
    if (prev.lastActiveDate) {
      const msPerDay = 24 * 60 * 60 * 1000;
      const diffDays = Math.round(
        (new Date(today).getTime() - new Date(prev.lastActiveDate).getTime()) / msPerDay,
      );
      if (diffDays === 1) newStreak = prev.counters.currentStreak + 1;
    }

    stateRef.current = {
      ...prev,
      lastActiveDate: today,
      counters: {
        ...prev.counters,
        currentStreak: newStreak,
        longestStreak: Math.max(prev.counters.longestStreak, newStreak),
      },
    };
    // eslint-disable-next-line react-hooks/set-state-in-effect -- daily bonus gated by date comparison
    awardXP('DAILY_LOGIN_BONUS');
  }, [isAuthenticated, hydrated, awardXP]);

  const unlockedIdSet = useMemo(
    () => new Set(state.unlockedAchievementIds),
    [state.unlockedAchievementIds],
  );
  const isAchievementUnlockedById = useCallback(
    (id: string) => unlockedIdSet.has(id),
    [unlockedIdSet],
  );

  const progress = useMemo(() => progressWithinLevel(state.xp), [state.xp]);

  const value = useMemo<GamificationContextType>(
    () => ({
      state,
      hydrated,
      level: progress.level,
      xpIntoLevel: progress.xpIntoLevel,
      xpForNext: progress.xpForNext,
      progressPercent: progress.percent,
      unlockedCount: state.unlockedAchievementIds.length,
      awardXP,
      isAchievementUnlocked: isAchievementUnlockedById,
    }),
    [state, hydrated, progress, awardXP, isAchievementUnlockedById],
  );

  return <GamificationContext.Provider value={value}>{children}</GamificationContext.Provider>;
}

export function useGamification() {
  const ctx = useContext(GamificationContext);
  if (!ctx) throw new Error('useGamification must be used within a GamificationProvider');
  return ctx;
}
