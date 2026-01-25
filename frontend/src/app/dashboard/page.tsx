"use client";

import { Layout } from "@/components/layout";
import { useTheme } from "@/providers";
import { usePlantsQuery } from "@/hooks/usePlants";
import { HomeScreen } from "@/components/screens/HomeScreen";

export default function DashboardPage() {
  const { theme, toggleTheme } = useTheme();
  const { data: plants = [] } = usePlantsQuery();
  return (
    <Layout darkMode={theme === "dark"} onToggleDarkMode={toggleTheme}>
      <HomeScreen darkMode={theme === "dark"} plants={plants} />
    </Layout>
  );
}
