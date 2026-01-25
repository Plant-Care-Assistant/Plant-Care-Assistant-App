"use client";

import { Layout } from "@/components/layout";
import { ProfileScreen } from "@/components/screens/ProfileScreen";
import { useTheme } from "@/providers";

import { ProtectedRoute } from "@/components/auth/ProtectedRoute";
// ...existing imports...

export default function ProfilePage() {
  const { theme } = useTheme();
  return (
    <Layout>
      <ProtectedRoute>
        <ProfileScreen darkMode={theme === 'dark'} />
      </ProtectedRoute>
    </Layout>
  );
}
