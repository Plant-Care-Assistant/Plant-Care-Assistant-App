"use client";

import { Layout } from "@/components/layout";
import { CollectionScreen } from "@/components/screens/CollectionScreen";
import { ProtectedRoute } from "@/components/auth/ProtectedRoute";
import { useTheme } from "@/providers";

export default function CollectionPage() {
  const { theme, toggleTheme } = useTheme();
  return (
    <Layout darkMode={theme === "dark"} onToggleDarkMode={toggleTheme}>
      <ProtectedRoute>
        <CollectionScreen darkMode={theme === "dark"} />
      </ProtectedRoute>
    </Layout>
  );
}
