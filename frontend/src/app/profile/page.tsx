"use client";

import { Layout } from "@/components/layout";
import { ProfileScreen } from "@/components/screens/ProfileScreen";
import { useTheme } from "@/providers";

import { ProtectedRoute } from "@/components/auth/ProtectedRoute";
// ...existing imports...

export default function ProfilePage() {
  const { theme, toggleTheme } = useTheme();
  return (
    <Layout darkMode={theme === "dark"} onToggleDarkMode={toggleTheme}>
      <ProtectedRoute>
        <ProfileScreen
          darkMode={theme === "dark"}
          onDarkModeToggle={toggleTheme}
        />
      </ProtectedRoute>
    </Layout>
  );
}
