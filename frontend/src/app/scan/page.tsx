"use client";

import { Layout } from "@/components/layout";
import { useState } from "react";
import { ProtectedRoute } from "@/components/auth/ProtectedRoute";
import { ScanScreen } from "@/components/screens/ScanScreen";
import { useTheme } from "@/providers";
import { useAddPlantMutation, usePlantsQuery } from "@/hooks/usePlants";
import type { Plant } from "@/lib/utils/plantFilters";
export default function ScanPage() {
  const { theme, toggleTheme } = useTheme();
  const { data: apiPlants = [] } = usePlantsQuery();
  const addPlantMutation = useAddPlantMutation();
  const [localAdditions, setLocalAdditions] = useState<Plant[]>([]);

  const plants = [...apiPlants, ...localAdditions];

  const handlePlantsChange = (newPlants: Plant[]) => {
    const existingIds = new Set([
      ...apiPlants.map((p) => p.id),
      ...localAdditions.map((p) => p.id),
    ]);
    const addedPlants = newPlants.filter((np) => !existingIds.has(np.id));

    if (addedPlants.length > 0) {
      setLocalAdditions((prev) => [...prev, ...addedPlants]);
      addedPlants.forEach((plant) => addPlantMutation.mutate(plant));
    }
  };

  return (
    <Layout darkMode={theme === "dark"} onToggleDarkMode={toggleTheme}>
      <ProtectedRoute>
        <ScanScreen
          darkMode={theme === "dark"}
          plants={plants}
          onPlantsChange={handlePlantsChange}
        />
      </ProtectedRoute>
    </Layout>
  );
}
