'use client';

import { useEffect } from 'react';
import { useGamification } from '@/providers';
import type { NavScreen } from '@/components/layout/navItems';
import type { XpActionId } from './xp-actions';

const FIRST_VISIT_ACTIONS: Record<NavScreen, XpActionId> = {
  home: 'FIRST_HOME_VISIT',
  scan: 'FIRST_SCAN_VISIT',
  collection: 'FIRST_COLLECTION_VISIT',
  profile: 'FIRST_PROFILE_VISIT',
};

export function useFirstVisitXP(screen: NavScreen) {
  const { hydrated, awardXP } = useGamification();
  useEffect(() => {
    if (!hydrated) return;
    awardXP(FIRST_VISIT_ACTIONS[screen]);
  }, [hydrated, screen, awardXP]);
}
