"use client";

import { Layout } from "@/components/layout";
import { ProfileScreen } from "@/components/screens/ProfileScreen";
import { useTheme } from "@/providers";

export default function ProfilePage() {
  const { theme, toggleTheme } = useTheme();

  return (
    <Layout
      showBottomNav
      showSidebar
      darkMode={theme === "dark"}
      onToggleDarkMode={toggleTheme}
    >
      <ProfileScreen darkMode={theme === "dark"} onDarkModeToggle={toggleTheme} />
    </Layout>
  );
}
