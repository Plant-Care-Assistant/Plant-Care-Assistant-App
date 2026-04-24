'use client';

import { Layout } from '@/components/layout';
import { HomeScreen } from '@/components/screens/HomeScreen';
import { usePlantsQuery } from '@/hooks/usePlants';
import { useTheme } from '@/providers';
import { useFirstVisitXP } from '@/lib/gamification/useFirstVisitXP';

export default function App() {
  const { toggleTheme } = useTheme();
  const { data: plants = [] } = usePlantsQuery();

  useFirstVisitXP('home');

  return (
    <Layout showBottomNav showSidebar onToggleDarkMode={toggleTheme}>
      <HomeScreen plants={plants} />
    </Layout>
  );
}
