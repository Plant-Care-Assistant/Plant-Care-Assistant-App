"use client";

import { ReactNode } from "react";
import { QueryProvider } from "./query-provider";
import { AuthProvider, useAuth } from "./auth-provider";
import { ThemeProvider, useTheme } from "./theme-provider";
import { ToastProvider, useToast } from "./toast-provider";
import { GamificationProvider, useGamification } from "./gamification-provider";

// GamificationProvider needs both AuthProvider (user id) and ToastProvider (toasts) to wrap it.
export function AppProviders({ children }: { children: ReactNode }) {
  return (
    <QueryProvider>
      <AuthProvider>
        <ToastProvider>
          <GamificationProvider>
            <ThemeProvider>{children}</ThemeProvider>
          </GamificationProvider>
        </ToastProvider>
      </AuthProvider>
    </QueryProvider>
  );
}

export { useAuth, useTheme, useToast, useGamification };
