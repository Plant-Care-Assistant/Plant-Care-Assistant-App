"use client";

import { Layout } from "@/components/layout";
import { ScanScreen } from "@/components/screens/ScanScreen";
import { useTheme } from "@/providers";
import { useFirstVisitXP } from "@/lib/gamification/useFirstVisitXP";

export default function ScanPage() {
  const { toggleTheme } = useTheme();

  useFirstVisitXP('scan');

  return (
    <Layout showBottomNav onToggleDarkMode={toggleTheme}>
      <ScanScreen />
    </Layout>
  );
}
