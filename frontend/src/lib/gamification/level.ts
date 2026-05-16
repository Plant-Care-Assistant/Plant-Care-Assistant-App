// XP to advance L->L+1 = 100 + 50*L; cumulative threshold to REACH level N = 25*(N-1)*(N+4).

export function xpToAdvance(level: number): number {
  return 100 + 50 * Math.max(1, level);
}

export function xpForLevelStart(level: number): number {
  if (level <= 1) return 0;
  return 25 * (level - 1) * (level + 4);
}

// Iterative for float-error robustness; bounded to 1000 as a guard.
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
