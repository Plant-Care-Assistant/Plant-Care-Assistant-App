'use client';

import { useEffect, useState } from 'react';

const CHALLENGES = [
  { description: 'Stay active 3 days in a row', target: 3 },
  { description: 'Maintain a 5-day care streak', target: 5 },
  { description: 'Keep your streak going for 7 days', target: 7 },
  { description: 'Build a 10-day care streak', target: 10 },
  { description: 'Stay consistent for 14 days', target: 14 },
  { description: 'Reach a 21-day care streak', target: 21 },
  { description: 'Hit a 30-day care streak', target: 30 },
];

const STORAGE_KEY = 'wc_idx';

export function useWeeklyChallenge(streak: number) {
  const [idx, setIdx] = useState<number>(() => {
    if (typeof window === 'undefined') return 0;
    const stored = parseInt(localStorage.getItem(STORAGE_KEY) ?? '0', 10) || 0;

    // Eagerly advance past challenges the user has already surpassed so we
    // never cascade through multiple state updates on mount. Cap at
    // CHALLENGES.length iterations to avoid an infinite loop when streak
    // exceeds every target (the list wraps and we stop at the first target
    // that is strictly greater than streak).
    let cur = stored;
    for (let i = 0; i < CHALLENGES.length; i++) {
      if (CHALLENGES[cur % CHALLENGES.length].target > streak) break;
      cur = (cur + 1) % CHALLENGES.length;
    }
    if (cur !== stored) localStorage.setItem(STORAGE_KEY, String(cur));
    return cur;
  });

  const challenge = CHALLENGES[idx % CHALLENGES.length];

  // Advance in real-time during a session if the streak crosses the target.
  useEffect(() => {
    if (streak >= challenge.target) {
      const next = (idx + 1) % CHALLENGES.length;
      localStorage.setItem(STORAGE_KEY, String(next));
      setIdx(next);
    }
  }, [streak, challenge.target, idx]);

  return {
    description: challenge.description,
    current: Math.min(streak, challenge.target),
    total: challenge.target,
  };
}
