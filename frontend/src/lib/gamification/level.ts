/**
 * Progressive level curve. Each level costs more XP than the previous one.
 *
 *   XP to advance from level N to N+1 = 100 + 50*N
 *     Lv 1 → 2: 150 XP
 *     Lv 2 → 3: 200 XP
 *     Lv 3 → 4: 250 XP
 *     Lv N → N+1: 100 + 50*N XP
 *
 *   Cumulative XP to REACH level N (i.e., XP threshold at which you become level N):
 *     cumulative(N) = 25 * (N - 1) * (N + 4)
 *     Lv 1  = 0
 *     Lv 2  = 150
 *     Lv 3  = 350
 *     Lv 5  = 900
 *     Lv 10 = 3150
 *     Lv 25 = 17400
 */

/** XP required to advance FROM `level` TO `level + 1`. */
export function xpToAdvance(level: number): number {
  return 100 + 50 * Math.max(1, level);
}

/** Cumulative XP threshold at which the user reaches `level`. */
export function xpForLevelStart(level: number): number {
  if (level <= 1) return 0;
  return 25 * (level - 1) * (level + 4);
}

/**
 * Resolve level from total XP. Iterative for robustness against float error.
 * Max ~50 iterations in practice; bounded to 1000 as a guard.
 */
export function levelFromXp(xp: number): number {
  if (xp < 0) return 1;
  let level = 1;
  let cumulative = 0;
  while (level < 1000) {
    const needed = xpToAdvance(level);
    if (cumulative + needed > xp) return level;
    cumulative += needed;
    level++;
  }
  return level;
}

export function progressWithinLevel(xp: number): {
  level: number;
  xpIntoLevel: number;
  xpForNext: number;
  percent: number;
} {
  const level = levelFromXp(xp);
  const base = xpForLevelStart(level);
  const xpIntoLevel = xp - base;
  const xpForNext = xpToAdvance(level);
  return {
    level,
    xpIntoLevel,
    xpForNext,
    percent: xpForNext === 0 ? 0 : (xpIntoLevel / xpForNext) * 100,
  };
}
