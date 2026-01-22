"use client";

import { ReactNode } from "react";
import { QueryProvider } from "./query-provider";
import { AuthProvider, useAuth } from "./auth-provider";
import { ThemeProvider, useTheme } from "./theme-provider";

/**
 * Root provider component that combines all providers
 * Order matters: QueryProvider -> AuthProvider -> ThemeProvider
 */
export function AppProviders({ children }: { children: ReactNode }) {
  return (
    <QueryProvider>
      <AuthProvider>
        <ThemeProvider>{children}</ThemeProvider>
      </AuthProvider>
    </QueryProvider>
  );
}

export { useAuth, useTheme };
