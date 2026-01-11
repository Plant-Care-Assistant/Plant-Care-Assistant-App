"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/providers";
import { cn } from "@/lib/utils/cn";

/**
 * Props for the AuthLayout wrapper.
 * @property children Page content.
 */
export interface AuthLayoutProps {
  children: React.ReactNode;
  className?: string;
}

/**
 * Layout for auth pages (login/signup).
 * Redirects to home if already authenticated.
 */
export function AuthLayout({ children, className }: AuthLayoutProps) {
  const router = useRouter();
  const { isAuthenticated } = useAuth();

  useEffect(() => {
    if (isAuthenticated) {
      router.push("/");
    }
  }, [isAuthenticated, router]);

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
