"use client";

import { useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Layout } from "@/components/layout";
import { DayStreakCard } from "@/components/features/plant/DayStreakCard";
import { StatCard } from "@/components/features/plant/StatCard";
import { WeeklyCare } from "@/components/features/plant/WeeklyCare";
import { EnvInfoCard } from "@/components/features/plant/EnvInfoCard";
import { CareInstructions } from "@/components/features/plant/CareInstructions";
import { PlantActions } from "@/components/features/plant/PlantActions";
import { useTheme, useGamification } from "@/providers";
import {
  usePlantQuery,
  useDeletePlantMutation,
  useCatalogPlantQuery,
  useCareHistoryQuery,
  useRecordWateringMutation,
  useRecordCareMutation,
} from "@/hooks/usePlants";
import type { CareType } from "@/types";
import { removePlantImage } from "@/lib/utils/plantImages";
import { Trash2, AlertCircle, CheckCircle2, Pencil } from "lucide-react";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { PlantImageGallery } from "@/components/features/plant/PlantImageGallery";
import { EditPlantDialog } from "@/components/features/plant/EditPlantDialog";
import { getDiseaseAdvice } from "@/lib/disease/advice";

export default function PlantDetailPage() {
  const params = useParams();
  const router = useRouter();
  const plantId = useMemo(() => Number(params?.id), [params]);
  const { data: plant, isLoading } = usePlantQuery(plantId || undefined);
  const { data: catalog } = useCatalogPlantQuery(plant?.plant_catalog_id);
  const { data: careHistory } = useCareHistoryQuery(plantId || undefined);
  const recordWateringMutation = useRecordWateringMutation(plantId || 0);
  const recordCareMutation = useRecordCareMutation(plantId || 0);
  const { theme, toggleTheme } = useTheme();
  const darkMode = theme === "dark";
  const { awardXP } = useGamification();
  const deletePlantMutation = useDeletePlantMutation();
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [showEditDialog, setShowEditDialog] = useState(false);

  // Care fields fall back through: per-plant override → catalog → mock default.
  // Mocks keep the cards informative when the species isn't in the catalog
  // (e.g. plant added without a successful AI scan).
  const tempMin = plant?.preferred_temp_min ?? catalog?.preferred_temp_min ?? 18;
  const tempMax = plant?.preferred_temp_max ?? catalog?.preferred_temp_max ?? 24;
  const tempLabel = `${tempMin}–${tempMax}°C`;

  const lightRaw = plant?.preferred_sunlight ?? catalog?.preferred_sunlight ?? "medium";
  const lightLabel = lightRaw[0].toUpperCase() + lightRaw.slice(1);

  const wateringIntervalDays =
    plant?.preferred_watering_interval_days
    ?? catalog?.preferred_watering_interval_days
    ?? 7;

  // Healthy % derives from the last AI verdict + its confidence.
  // Confidence is stored inconsistently (sometimes 0..1 fraction, sometimes 1..100 percent)
  // so normalize defensively. Diseased verdicts invert the confidence so that a
  // high-confidence disease detection reads as low health.
  const healthPercent = (() => {
    if (!plant?.last_health_label) return 70;
    const raw = plant.last_health_confidence;
    if (raw == null) return plant.last_health_label === "healthy" ? 90 : 35;
    const confPct = raw > 1 ? Math.min(100, Math.round(raw)) : Math.round(raw * 100);
    return plant.last_health_label === "healthy" ? confPct : 100 - confPct;
  })();

  const streakDays = careHistory?.current_streak_days ?? 0;
  const weeklyActiveDays = careHistory?.unique_days_last_week ?? 0;
  // Backend returns oldest→today; fall back to an empty 7-day strip ending
  // today so the grid still renders before the query resolves.
  const dailyLastWeek = careHistory?.daily_last_week ?? (() => {
    const out: { date: string; types: CareType[] }[] = [];
    const today = new Date();
    for (let i = 6; i >= 0; i--) {
      const d = new Date(today);
      d.setDate(today.getDate() - i);
      out.push({
        date: `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`,
        types: [],
      });
    }
    return out;
  })();
  const lastWateredAt = careHistory?.waterings?.[0]
    ? new Date(careHistory.waterings[0])
    : null;
  const lastWateredLabel = lastWateredAt ? lastWateredAt.toLocaleDateString() : "—";

  // Day-cycle card now shows days remaining until the next watering is due.
  // Falls back to the full interval when the plant has never been watered.
  const daysSinceLastWater = lastWateredAt
    ? Math.floor((Date.now() - lastWateredAt.getTime()) / 86_400_000)
    : null;
  const daysUntilNextWater =
    daysSinceLastWater == null
      ? wateringIntervalDays
      : Math.max(0, wateringIntervalDays - daysSinceLastWater);
  const cycleLabel = daysUntilNextWater === 0 ? "water now" : "days to water";

  const handleWaterNow = () => {
    const plantName = plant?.custom_name || 'Your plant';
    recordWateringMutation.mutate();
    awardXP('WATER_PLANT', { subtitle: plantName });
    if (new Date().getHours() < 9) {
      awardXP('WATER_BEFORE_9AM', { subtitle: 'Watered before 9 AM' });
    }
  };

  const handleLogCare = (type: Exclude<CareType, 'water'>) => {
    const plantName = plant?.custom_name || 'Your plant';
    recordCareMutation.mutate(type);
    awardXP('COMPLETE_CARE_TASK', { subtitle: `${plantName} — ${type}` });
  };

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
          {/* Hero = photo gallery: latest (50% width) + 2 previous (right column).
              Plant name, species, health badge, and back button overlay on the latest tile. */}
          {plant && (
            <PlantImageGallery
              plantId={plant.id}
              plantName={plant.custom_name || "Unknown Plant"}
              plantSpecies={plant.note || undefined}
              healthPercent={healthPercent}
              onBack={() => router.push('/collection')}
              darkMode={darkMode}
            />
          )}

          {/* Overdue watering banner — only when due today */}
          {plant && daysUntilNextWater === 0 && (
            <button
              onClick={handleWaterNow}
              className="w-full rounded-2xl p-4 flex items-center justify-between gap-3 border bg-red-50 border-red-200 dark:bg-red-900/20 dark:border-red-800 hover:opacity-90 transition-opacity text-left"
            >
              <div className="flex items-center gap-3">
                <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
                <div>
                  <p className={`font-semibold text-sm ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
                    Time to water this plant
                  </p>
                  <p className={`text-xs ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
                    {lastWateredAt
                      ? `Last watered ${lastWateredLabel}`
                      : 'No watering recorded yet'}
                  </p>
                </div>
              </div>
              <span className="text-sm font-semibold text-red-600 dark:text-red-300 shrink-0">
                Water now →
              </span>
            </button>
          )}

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
                  {plant.last_health_confidence != null && (
                    <span
                      className={`text-sm font-semibold ${
                        plant.last_health_label === 'diseased'
                          ? 'text-red-600 dark:text-red-300'
                          : 'text-green-600 dark:text-green-300'
                      }`}
                    >
                      {/* Stored as percent for new rows, but legacy rows used
                          fractions (0..1). Normalize defensively. */}
                      {plant.last_health_confidence > 1
                        ? Math.round(plant.last_health_confidence)
                        : Math.round(plant.last_health_confidence * 100)}
                      %
                    </span>
                  )}
                </div>
                {plant.last_health_check_at && (
                  <span className={`text-xs ${darkMode ? 'text-neutral-400' : 'text-neutral-500'}`}>
                    Last checked {new Date(plant.last_health_check_at).toLocaleDateString()}
                  </span>
                )}
              </div>

              {plant.last_health_label === 'diseased' && plant.last_diseases?.length ? (
                <>
                  <ul className="mt-1 space-y-1">
                    {plant.last_diseases.slice(0, 3).map((d, i) => (
                      <li key={i} className="flex justify-between text-xs">
                        <span className={darkMode ? 'text-neutral-300' : 'text-neutral-700'}>
                          {/* AI labels prefix with a training-set host species
                              (Apple/Orange/...) that doesn't match the user's
                              actual plant. Show only the condition. */}
                          {d.condition || d.plant}
                        </span>
                        <span className="text-neutral-400 ml-2 shrink-0">
                          {Math.round(d.confidence * 100)}%
                        </span>
                      </li>
                    ))}
                  </ul>
                  {(() => {
                    const top = plant.last_diseases[0];
                    const advice = getDiseaseAdvice(top.condition || top.plant);
                    if (!advice) return null;
                    return (
                      <div className="mt-3 pt-3 border-t border-red-200 dark:border-red-800">
                        <p className={`text-xs font-semibold mb-1 ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
                          What to do:
                        </p>
                        <ul className={`text-xs space-y-1 list-disc pl-4 ${
                          darkMode ? 'text-neutral-300' : 'text-neutral-700'
                        }`}>
                          {advice.map((tip, i) => <li key={i}>{tip}</li>)}
                        </ul>
                      </div>
                    );
                  })()}
                </>
              ) : null}
            </div>
          )}

          {/* Two-column layout on laptop */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 lg:gap-4">
            {/* Main column */}
            <div className="lg:col-span-8 space-y-3 sm:space-y-4">
              <DayStreakCard days={streakDays} />

              {/* Stats row */}
              <div className="grid grid-cols-3 gap-2 sm:gap-3">
                <StatCard type="watered" value={lastWateredLabel} darkMode={darkMode} />
                <StatCard
                  type="cycle"
                  value={daysUntilNextWater}
                  label={cycleLabel}
                  urgent={daysUntilNextWater === 0}
                  darkMode={darkMode}
                />
                <StatCard type="health" value={healthPercent} darkMode={darkMode} />
              </div>

              {/* Weekly care */}
              <WeeklyCare
                daily={dailyLastWeek}
                activeDays={weeklyActiveDays}
                darkMode={darkMode}
              />
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
                onLogCare={handleLogCare}
                darkMode={darkMode}
              />
            </div>
          </div>

          {/* Edit + Delete buttons */}
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => setShowEditDialog(true)}
              className={`py-4 rounded-2xl font-semibold transition-colors flex items-center justify-center gap-2 ${
                darkMode
                  ? 'text-secondary hover:bg-neutral-800'
                  : 'text-secondary hover:bg-secondary/10'
              }`}
            >
              <Pencil size={18} />
              Edit plant
            </button>
            <button
              onClick={() => setShowDeleteDialog(true)}
              className={`py-4 rounded-2xl font-semibold transition-colors flex items-center justify-center gap-2 ${
                darkMode
                  ? 'text-accent2 hover:bg-neutral-800'
                  : 'text-accent2 hover:bg-pink-50'
              }`}
            >
              <Trash2 size={18} />
              Remove from Collection
            </button>
          </div>

          {plant && (
            <EditPlantDialog
              plant={plant}
              open={showEditDialog}
              onOpenChange={setShowEditDialog}
            />
          )}

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
