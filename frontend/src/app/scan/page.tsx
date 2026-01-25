"use client";

import { Layout } from "@/components/layout";
import { Button, Input } from "@/components/ui";
import { Upload } from "lucide-react";
import { useState } from "react";
import { useIdentifyPlantMutation } from "@/hooks/usePlants";

import { ProtectedRoute } from "@/components/auth/ProtectedRoute";
// ...existing imports...

import { ScanScreen } from '@/components/screens/ScanScreen';
import { useTheme } from '@/providers';
import { usePlantsQuery } from '@/hooks/usePlants';
import { Plant as ScanPlant } from '@/lib/utils/plantFilters';


export default function ScanPage() {
  const { theme } = useTheme();
  const { data: apiPlants = [] } = usePlantsQuery();
  // Map API plants to ScanScreen's expected type
  const plants: ScanPlant[] = apiPlants.map((p) => ({
    id: p.id, // id is now number everywhere
    name: p.name,
    species: p.scientificName,
    lastWatered: p.lastWatered,
    health:
      p.health >= 80 ? 'healthy'
      : p.health >= 50 ? 'needs-attention'
      : 'critical',
    imageUrl: p.image,
    wateringFrequency: p.wateringFrequency,
    // Add more mappings as needed
  }));
  // No-op handler for onPlantsChange (scan does not mutate collection directly)
  return (
    <Layout>
      <ProtectedRoute>
        <ScanScreen darkMode={theme === 'dark'} plants={plants} onPlantsChange={() => {}} />
      </ProtectedRoute>
    </Layout>
  );
}
