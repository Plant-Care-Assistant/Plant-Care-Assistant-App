'use client';

import { useEffect } from 'react';
import { Layout } from '@/components/layout';
import { HomeScreen } from '@/components/screens/HomeScreen';
import { usePlantsQuery } from '@/hooks/usePlants';
import { useTheme } from '@/providers';
import { useFirstVisitXP } from '@/lib/gamification/useFirstVisitXP';
import { maybeSendCareReminder } from '@/lib/notifications/careReminders';

export default function App() {
  const { toggleTheme } = useTheme();
  const { data: plants = [], isSuccess } = usePlantsQuery();

  useFirstVisitXP('home');

  // Fire once-per-day reminder when plants first load. The dependency is
  // isSuccess only — maybeSendCareReminder guards internally against re-firing
  // within the same calendar day, so stale plant references are safe.
  useEffect(() => {
    if (!isSuccess) return;
    const overdue = plants.filter(
      (p) => p.days_until_water != null && p.days_until_water <= 0,
    );
    maybeSendCareReminder(overdue);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isSuccess]);

  return (
    <Layout showBottomNav showSidebar onToggleDarkMode={toggleTheme}>
      <HomeScreen plants={plants} />
    </Layout>
  );
}
