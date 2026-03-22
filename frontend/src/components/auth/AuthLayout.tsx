"use client";

import { cn } from "@/lib/utils/cn";

export interface AuthLayoutProps {
  children: React.ReactNode;
  className?: string;
}

/**
 * Layout for auth pages (login/signup).
 * Route protection is handled by middleware.
 */
export function AuthLayout({ children, className }: AuthLayoutProps) {
  return (
    <div
      className={cn(
        "min-h-screen bg-neutral-50 text-neutral-900 dark:bg-neutral-950 dark:text-neutral-50 flex items-center justify-center px-4",
        className,
      )}
    >
      <div className="w-full max-w-sm">{children}</div>
    </div>
  );
}

export default AuthLayout;
