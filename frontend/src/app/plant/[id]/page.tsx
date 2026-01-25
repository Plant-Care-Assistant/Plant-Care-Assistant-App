"use client";

import { useMemo } from "react";
import { useParams, useRouter } from "next/navigation";
import { Layout } from "@/components/layout";
import { PlantHero } from "@/components/features/plant/PlantHero";
import { DayStreakCard } from "@/components/features/plant/DayStreakCard";
import { StatCard } from "@/components/features/plant/StatCard";
import { WeeklyCare } from "@/components/features/plant/WeeklyCare";
import { EnvInfoCard } from "@/components/features/plant/EnvInfoCard";
import { CareInstructions } from "@/components/features/plant/CareInstructions";
import { PlantActions } from "@/components/features/plant/PlantActions";
import { useTheme } from "@/providers";
import { usePlantQuery } from "@/hooks/usePlants";

export default function PlantDetailPage() {
  const params = useParams();
  const router = useRouter();
  const plantId = useMemo(() => params?.id?.toString() ?? "", [params]);
  const { data: plant, isLoading } = usePlantQuery(plantId);
  const { theme, toggleTheme } = useTheme();

  return (
    <Layout
      showBottomNav
      showSidebar
      darkMode={theme === "dark"}
      onToggleDarkMode={toggleTheme}
    >
      <div className="p-3 sm:p-4 lg:p-6 pb-28 sm:pb-24 lg:pb-4 max-w-7xl mx-auto">
        <div className="space-y-4 lg:space-y-6">
          {/* Hero spans full width */}
          <PlantHero
            name={plant?.name || "Unknown Plant"}
            species={plant?.species}
            imageUrl={plant?.imageUrl}
            healthPercent={85}
            onBack={() => router.push("/collection")}
            darkMode={theme === "dark"}
          />

          {/* Two-column layout on laptop */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 lg:gap-4">
            {/* Main column */}
            <div className="lg:col-span-8 space-y-3 sm:space-y-4">
              <DayStreakCard days={8} />

              {/* Stats row */}
              <div className="grid grid-cols-3 gap-2 sm:gap-3">
                <StatCard
                  type="watered"
                  value={plant?.lastWatered || "2 days ago"}
                  darkMode={theme === "dark"}
                />
                <StatCard type="cycle" value={5} darkMode={theme === "dark"} />
                <StatCard
                  type="health"
                  value={85}
                  darkMode={theme === "dark"}
                />
              </div>

              {/* Weekly care */}
              <WeeklyCare
                totalDays={7}
                activeDays={3}
                darkMode={theme === "dark"}
              />
            </div>

            {/* Sidebar */}
            <div className="lg:col-span-4 space-y-3 sm:space-y-4 lg:space-y-6 lg:sticky lg:top-24">
              {/* Environment info */}
              <div className="grid grid-cols-2 lg:grid-cols-1 gap-2 sm:gap-3">
                <EnvInfoCard
                  type="temperature"
                  value="18-26°C"
                  darkMode={theme === "dark"}
                />
                <EnvInfoCard
                  type="light"
                  value="Low to medium"
                  darkMode={theme === "dark"}
                />
              </div>

              {/* Care instructions */}
              <CareInstructions
                items={[
                  "Keep soil consistently moist",
                  "Tolerates low light well",
                  "Mist leaves regularly",
                  "Remove brown leaf tips",
                ]}
                darkMode={theme === "dark"}
              />

              {/* Actions */}
              <PlantActions
                onWaterNow={() => console.log("Water now", plantId)}
                onGainXP={() => console.log("Gain XP", plantId)}
                darkMode={theme === "dark"}
              />
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}
