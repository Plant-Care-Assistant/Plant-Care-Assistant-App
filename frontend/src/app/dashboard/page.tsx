"use client";

import { Layout } from "@/components/layout";
import { useTheme } from "@/providers";
import { usePlantsQuery } from "@/hooks/usePlants";
import { HomeScreen } from "@/components/screens/HomeScreen";

export default function DashboardPage() {
  const { theme } = useTheme();
  const { data: plants = [] } = usePlantsQuery();
  return (
    <Layout>
      <HomeScreen darkMode={theme === 'dark'} plants={plants} />
    </Layout>
  );
}
