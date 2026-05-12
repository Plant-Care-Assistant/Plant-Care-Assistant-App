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
import { ACHIEVEMENTS } from '@/lib/data/achievements';
import {
  gamificationApi,
  type NormalizedSnapshot,
} from '@/lib/api/gamification';

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

function snapshotToState(snap: NormalizedSnapshot): GamificationState {
  return {
    xp: snap.xp,
    counters: snap.counters,
    flags: snap.flags,
    unlockedAchievementIds: snap.unlockedAchievementIds,
    lastActiveDate: snap.lastActiveDate,
  };
}

/** One-time cleanup: drop stale `gamification:*` localStorage keys from v1. */
function purgeLegacyLocalStorage() {
  if (typeof window === 'undefined') return;
  try {
    const keys = Object.keys(window.localStorage).filter((k) => k.startsWith('gamification:'));
    keys.forEach((k) => window.localStorage.removeItem(k));
  } catch {
    /* ignore */
  }
}

export function GamificationProvider({ children }: { children: ReactNode }) {
  const { isAuthenticated, isLoading: authLoading } = useAuth();
  const { showXpToast, showAchievementToast } = useToast();
  const [state, setState] = useState<GamificationState>(EMPTY_STATE);
  const [hydrated, setHydrated] = useState(false);
  const stateRef = useRef<GamificationState>(EMPTY_STATE);

  // Hydrate from backend on auth ready. Legacy localStorage is purged once.
  useEffect(() => {
    if (authLoading) return;
    if (!isAuthenticated) {
      setState(EMPTY_STATE);
      stateRef.current = EMPTY_STATE;
      setHydrated(true);
      return;
    }
    let cancelled = false;
    (async () => {
      purgeLegacyLocalStorage();
      const snap = await gamificationApi.getMe();
      if (cancelled) return;
      const next = snap ? snapshotToState(snap) : EMPTY_STATE;
      stateRef.current = next;
      setState(next);
      setHydrated(true);
    })();
    return () => {
      cancelled = true;
    };
  }, [authLoading, isAuthenticated]);

  const awardXP = useCallback<GamificationContextType['awardXP']>(
    (actionId, opts) => {
      const action = XP_ACTIONS[actionId];
      if (!action) return;

      const prev = stateRef.current;
      // Once-only actions short-circuit when the flag is already set, avoiding
      // a needless round-trip. Backend is also idempotent here (§7.2) — this is
      // just an optimization, not a correctness guard.
      if (action.onceOnly && action.flag && prev.flags[action.flag]) return;

      // Fire-and-forget: backend is authoritative. We apply the returned
      // snapshot. Toasts are driven by `xp_awarded` / `newly_unlocked`.
      (async () => {
        const res = await gamificationApi.postEvent(actionId);
        if (!res) return;

        stateRef.current = snapshotToState(res.snapshot);
        setState(stateRef.current);

        if (res.xpAwarded > 0) {
          // Subtract the achievement-unlock bonus from the main toast so the
          // numbers add up cleanly (main action toast + per-unlock bonus toast).
          const unlockBonus =
            res.newlyUnlocked.length * XP_ACTIONS.ACHIEVEMENT_UNLOCK.xp;
          const actionXp = res.xpAwarded - unlockBonus;
          if (actionXp > 0) {
            showXpToast({
              amount: actionXp,
              title: action.label,
              subtitle: opts?.subtitle ?? action.description,
            });
          }
          if (unlockBonus > 0) {
            showXpToast({
              amount: unlockBonus,
              title: XP_ACTIONS.ACHIEVEMENT_UNLOCK.label,
              subtitle: `${res.newlyUnlocked.length} achievement${
                res.newlyUnlocked.length > 1 ? 's' : ''
              } unlocked`,
            });
          }
        }
        res.newlyUnlocked.forEach((id) => {
          const a = ACHIEVEMENTS.find((x) => x.id === id);
          if (a) showAchievementToast({ title: a.title, subtitle: a.hint });
        });
      })();
    },
    [showXpToast, showAchievementToast],
  );

  // First login: fire once per session when authenticated. Backend's
  // GameAction.first_login is gated by the `first_login` flag (§7.2) so any
  // duplicate fires return xp_awarded=0 with an unchanged snapshot.
  useEffect(() => {
    if (!isAuthenticated || !hydrated) return;
    if (!state.flags.accountCreated) awardXP('FIRST_LOGIN');
  }, [isAuthenticated, hydrated, state.flags.accountCreated, awardXP]);

  // Daily login: backend handles the once-per-day gate via last_login_at, so
  // we can fire on every mount and trust the no-op when already awarded today.
  useEffect(() => {
    if (!isAuthenticated || !hydrated) return;
    awardXP('DAILY_LOGIN_BONUS');
    // Intentionally fires once on hydration; backend dedupes by calendar day.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated, hydrated]);

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
