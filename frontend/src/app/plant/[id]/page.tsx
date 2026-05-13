"use client";

import { useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Layout } from "@/components/layout";
import { PlantHero } from "@/components/features/plant/PlantHero";
import { DayStreakCard } from "@/components/features/plant/DayStreakCard";
import { StatCard } from "@/components/features/plant/StatCard";
import { WeeklyCare } from "@/components/features/plant/WeeklyCare";
import { EnvInfoCard } from "@/components/features/plant/EnvInfoCard";
import { CareInstructions } from "@/components/features/plant/CareInstructions";
import { PlantActions } from "@/components/features/plant/PlantActions";
import { useTheme, useGamification } from "@/providers";
import { usePlantQuery, useDeletePlantMutation, useCatalogPlantQuery } from "@/hooks/usePlants";
import { getPlantImage } from "@/lib/utils/plantImages";
import { removePlantImage } from "@/lib/utils/plantImages";
import { Trash2 } from "lucide-react";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { AlertCircle, CheckCircle2 } from "lucide-react";

export default function PlantDetailPage() {
  const params = useParams();
  const router = useRouter();
  const plantId = useMemo(() => Number(params?.id), [params]);
  const { data: plant, isLoading } = usePlantQuery(plantId || undefined);
  const { data: catalog } = useCatalogPlantQuery(plant?.plant_catalog_id);
  const { theme, toggleTheme } = useTheme();
  const darkMode = theme === "dark";
  const { awardXP } = useGamification();
  const deletePlantMutation = useDeletePlantMutation();
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const tempLabel =
    catalog?.preferred_temp_min != null && catalog?.preferred_temp_max != null
      ? `${catalog.preferred_temp_min}–${catalog.preferred_temp_max}°C`
      : "—";
  const lightLabel = catalog?.preferred_sunlight
    ? catalog.preferred_sunlight[0].toUpperCase() + catalog.preferred_sunlight.slice(1)
    : "—";
  const wateringValue: string | number =
    catalog?.preferred_watering_interval_days ?? "—";

  const handleWaterNow = () => {
    const plantName = plant?.custom_name || 'Your plant';
    awardXP('WATER_PLANT', { subtitle: plantName });
    if (new Date().getHours() < 9) {
      awardXP('WATER_BEFORE_9AM', { subtitle: 'Watered before 9 AM' });
    }
  };

  const handleGainXP = () => {
    const plantName = plant?.custom_name || 'Your plant';
    awardXP('COMPLETE_CARE_TASK', { subtitle: plantName });
  };

  const imageUrl = useMemo(() => plantId ? getPlantImage(plantId) : undefined, [plantId]);

  const handleDelete = async () => {
    if (!plant) return;
    try {
      await deletePlantMutation.mutateAsync(plant.id);
      removePlantImage(plant.id);
      awardXP('DELETE_PLANT', { subtitle: plant.custom_name ?? 'Plant' });
      setShowDeleteDialog(false);
      router.push("/collection");
    } catch {
      setShowDeleteDialog(false);
    }
  };

  if (isLoading) {
    return (
      <Layout showBottomNav showSidebar onToggleDarkMode={toggleTheme}>
        <div className="p-6 text-center">Loading...</div>
      </Layout>
    );
  }

  return (
    <Layout
      showBottomNav
      showSidebar
      onToggleDarkMode={toggleTheme}
    >
      <div className="p-3 sm:p-4 lg:p-6 pb-28 sm:pb-24 lg:pb-4 max-w-7xl mx-auto">
        <div className="space-y-4 lg:space-y-6">
          {/* Hero spans full width */}
          <PlantHero
            name={plant?.custom_name || "Unknown Plant"}
            species={plant?.note || undefined}
            imageUrl={imageUrl}
            healthPercent={85}
            onBack={() => router.push('/collection')}
            darkMode={darkMode}
          />

          {/* Health verdict from last AI scan (if any) */}
          {plant?.last_health_label && (
            <div
              className={`rounded-2xl p-4 flex flex-col gap-2 border ${
                plant.last_health_label === 'diseased'
                  ? 'bg-red-50 border-red-200 dark:bg-red-900/20 dark:border-red-800'
                  : 'bg-green-50 border-green-200 dark:bg-green-900/20 dark:border-green-800'
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  {plant.last_health_label === 'diseased' ? (
                    <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
                  ) : (
                    <CheckCircle2 className="h-5 w-5 text-green-500 flex-shrink-0" />
                  )}
                  <span
                    className={`font-semibold text-sm ${
                      darkMode ? 'text-white' : 'text-neutral-900'
                    }`}
                  >
                    {plant.last_health_label === 'diseased'
                      ? 'Plant may be diseased'
                      : 'Plant looks healthy'}
                  </span>
                </div>
                {plant.last_health_check_at && (
                  <span className={`text-xs ${darkMode ? 'text-neutral-400' : 'text-neutral-500'}`}>
                    Last checked {new Date(plant.last_health_check_at).toLocaleDateString()}
                  </span>
                )}
              </div>

              {plant.last_health_label === 'diseased' && plant.last_diseases?.length ? (
                <ul className="mt-1 space-y-1">
                  {plant.last_diseases.slice(0, 3).map((d, i) => (
                    <li key={i} className="flex justify-between text-xs">
                      <span className={darkMode ? 'text-neutral-300' : 'text-neutral-700'}>
                        {d.plant}
                        {d.condition ? ` — ${d.condition}` : ''}
                      </span>
                      <span className="text-neutral-400 ml-2 shrink-0">
                        {Math.round(d.confidence * 100)}%
                      </span>
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>
          )}

          {/* Two-column layout on laptop */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 lg:gap-4">
            {/* Main column */}
            <div className="lg:col-span-8 space-y-3 sm:space-y-4">
              <DayStreakCard days={8} />

              {/* Stats row */}
              <div className="grid grid-cols-3 gap-2 sm:gap-3">
                <StatCard type="watered" value="—" darkMode={darkMode} />
                <StatCard type="cycle" value={wateringValue} darkMode={darkMode} />
                <StatCard type="health" value={85} darkMode={darkMode} />
              </div>

              {/* Weekly care */}
              <WeeklyCare totalDays={7} activeDays={3} darkMode={darkMode} />
            </div>

            {/* Sidebar */}
            <div className="lg:col-span-4 space-y-3 sm:space-y-4 lg:space-y-6 lg:sticky lg:top-24">
              {/* Environment info */}
              <div className="grid grid-cols-2 lg:grid-cols-1 gap-2 sm:gap-3">
                <EnvInfoCard type="temperature" value={tempLabel} darkMode={darkMode} />
                <EnvInfoCard type="light" value={lightLabel} darkMode={darkMode} />
              </div>

              {/* Care instructions */}
              <CareInstructions
                items={[
                  'Keep soil consistently moist',
                  'Tolerates low light well',
                  'Mist leaves regularly',
                  'Remove brown leaf tips',
                ]}
                darkMode={darkMode}
              />

              {/* Actions */}
              <PlantActions
                onWaterNow={handleWaterNow}
                onGainXP={handleGainXP}
                darkMode={darkMode}
              />
            </div>
          </div>

          {/* Delete Button */}
          <button
            onClick={() => setShowDeleteDialog(true)}
            className={`w-full py-4 rounded-2xl font-semibold transition-colors flex items-center justify-center gap-2 ${
              darkMode
                ? 'text-accent2 hover:bg-neutral-800'
                : 'text-accent2 hover:bg-pink-50'
            }`}
          >
            <Trash2 size={18} />
            Remove from Collection
          </button>

          {/* Delete Confirmation Dialog */}
          <ConfirmDialog
            open={showDeleteDialog}
            onOpenChange={setShowDeleteDialog}
            title="Remove Plant"
            description={`Are you sure you want to remove "${plant?.custom_name || 'this plant'}" from your collection? This action cannot be undone.`}
            confirmLabel="Remove"
            cancelLabel="Keep"
            onConfirm={handleDelete}
            isLoading={deletePlantMutation.isPending}
          />
        </div>
      </div>
    </Layout>
  );
}
