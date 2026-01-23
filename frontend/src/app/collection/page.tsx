"use client";

import { Layout } from "@/components/layout";
import { Skeleton } from "@/components/ui";
import { CollectionScreen } from "@/components/screens/CollectionScreen";
import { useTheme } from "@/providers";
import { useState, useEffect } from "react";
import type { Plant } from "@/lib/utils/plantFilters";
import { MOCK_PLANTS } from "@/lib/utils/plantFilters";

export default function CollectionPage() {
  const { theme, toggleTheme } = useTheme();
  const darkMode = theme === "dark";
  const [plants, setPlants] = useState<Plant[]>(MOCK_PLANTS);
  const [isLoading, setIsLoading] = useState(false);

  // Simulate initial load (remove API call for instant load)
  useEffect(() => {
    // Use mock plants directly - no API call needed
    setPlants(MOCK_PLANTS);
    setIsLoading(false);
  }, []);

  return (
    <Layout 
      showBottomNav 
      darkMode={darkMode}
      onToggleDarkMode={toggleTheme}
    >
      {isLoading ? (
        <div className={`min-h-screen pb-24 lg:pb-8 ${darkMode ? "text-white" : "text-gray-900"}`}>
          <div className="p-4 lg:p-6 max-w-7xl mx-auto">
            {/* Header Skeleton */}
            <div className="mb-6">
              <Skeleton className="h-8 w-48 mb-2" />
              <Skeleton className="h-4 w-64" />
            </div>

            {/* Search Skeleton */}
            <Skeleton className="h-10 w-full mb-6 rounded-lg" />

            {/* Filters Skeleton */}
            <div className="flex gap-2 mb-6">
              {[1, 2, 3, 4].map((i) => (
                <Skeleton key={i} className="h-10 w-20 rounded-full" />
              ))}
            </div>

            {/* Plant Grid Skeleton */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 lg:gap-6">
              {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
                <div key={i} className={`rounded-2xl overflow-hidden ${darkMode ? "bg-neutral-800" : "bg-white"}`}>
                  {/* Image Skeleton */}
                  <Skeleton className="h-48 w-full" />
                  
                  {/* Content Skeleton */}
                  <div className="p-4 space-y-3">
                    <Skeleton className="h-5 w-32" />
                    <Skeleton className="h-4 w-24" />
                    <div className="space-y-2">
                      <Skeleton className="h-3 w-full" />
                      <Skeleton className="h-3 w-3/4" />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <CollectionScreen
          darkMode={darkMode}
          plants={plants}
          onPlantsChange={setPlants}
        />
      )}
    </Layout>
  );
}
