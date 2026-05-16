const ICON_COUNT = 8;

/**
 * Returns the plant icon path for a given user level, cycling through
 * ic-1.png … ic-8.png. Guards against level 0 (pre-hydration state).
 */
export function iconForLevel(level: number): string {
  const safe = Math.max(1, level);
  return `/ic-${((safe - 1) % ICON_COUNT) + 1}.png`;
}
