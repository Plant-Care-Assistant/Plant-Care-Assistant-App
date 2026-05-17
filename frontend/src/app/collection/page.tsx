"use client";

import { Layout } from "@/components/layout";
import { Skeleton } from "@/components/ui";
import { CollectionScreen } from "@/components/screens/CollectionScreen";
import { useTheme } from "@/providers";
import { usePlantsQuery } from "@/hooks/usePlants";
import { useFirstVisitXP } from "@/lib/gamification/useFirstVisitXP";

export default function CollectionPage() {
  const { toggleTheme } = useTheme();
  const { data: plants = [], isLoading } = usePlantsQuery();

  useFirstVisitXP('collection');

  return (
    <Layout showBottomNav onToggleDarkMode={toggleTheme}>
      {isLoading ? (
        <div className="min-h-screen pb-24 lg:pb-8 text-gray-900 dark:text-white">
          <div className="p-4 lg:p-6 max-w-7xl mx-auto">
            <div className="mb-6">
              <Skeleton className="h-8 w-48 mb-2" />
              <Skeleton className="h-4 w-64" />
            </div>

            <Skeleton className="h-10 w-full mb-6 rounded-lg" />

            <div className="flex gap-2 mb-6">
              {[1, 2, 3, 4].map((i) => (
                <Skeleton key={i} className="h-10 w-20 rounded-full" />
              ))}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 lg:gap-6">
              {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
                <div key={i} className="rounded-2xl overflow-hidden bg-white dark:bg-neutral-800">
                  <Skeleton className="h-48 w-full" />
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
        <CollectionScreen plants={plants} />
      )}
    </Layout>
  );
}
