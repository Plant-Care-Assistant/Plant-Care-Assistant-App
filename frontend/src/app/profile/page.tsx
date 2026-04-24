"use client";

import { Layout } from "@/components/layout";
import { ProfileScreen } from "@/components/screens/ProfileScreen";
import { useTheme } from "@/providers";
import { useFirstVisitXP } from "@/lib/gamification/useFirstVisitXP";

export default function ProfilePage() {
  const { toggleTheme } = useTheme();

  useFirstVisitXP('profile');

  return (
    <Layout showBottomNav showSidebar onToggleDarkMode={toggleTheme}>
      <ProfileScreen onDarkModeToggle={toggleTheme} />
    </Layout>
  );
}
