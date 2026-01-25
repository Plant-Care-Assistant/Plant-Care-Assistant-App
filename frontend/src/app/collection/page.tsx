"use client";

import { Layout } from "@/components/layout";
import { Skeleton } from "@/components/ui";
import { CollectionScreen } from "@/components/screens/CollectionScreen";
// import { useTheme } from "@/providers";
import { useState, useEffect } from "react";
import type { Plant } from "@/lib/utils/plantFilters";
import { MOCK_PLANTS } from "@/lib/utils/plantFilters";

import { ProtectedRoute } from "@/components/auth/ProtectedRoute";
// ...existing imports...

import { useTheme } from '@/providers';

export default function CollectionPage() {
  const { theme } = useTheme();
  return (
    <Layout>
      <ProtectedRoute>
        <CollectionScreen darkMode={theme === 'dark'} />
      </ProtectedRoute>
    </Layout>
  );
}
