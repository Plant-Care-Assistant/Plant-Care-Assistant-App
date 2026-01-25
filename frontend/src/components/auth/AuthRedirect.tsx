"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/providers";

export function AuthRedirect({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading, error } = useAuth();
  const [failCount, setFailCount] = useState(0);
  const lastLoading = useRef(false);
  const router = useRouter();

  useEffect(() => {
    // Only redirect if definitely authenticated and not loading
    if (!isLoading && isAuthenticated) {
      const redirectTo = sessionStorage.getItem("redirectAfterLogin") || "/dashboard";
      sessionStorage.removeItem("redirectAfterLogin");
      router.replace(redirectTo);
    }
  }, [isAuthenticated, isLoading, router]);

  // Track repeated loading flickers (backend unreachable)
  useEffect(() => {
    if (isLoading && !lastLoading.current) {
      setFailCount((c) => c + 1);
    }
    lastLoading.current = isLoading;
  }, [isLoading]);

  if (isLoading) {
    // If loading flickers too many times, show error
    if (failCount > 5) {
      return (
        <div className="flex h-screen w-full flex-col items-center justify-center text-center">
          <div className="mb-4 text-lg font-bold text-red-600">Unable to connect to authentication service.</div>
          <div className="mb-2 text-neutral-500">Please check your backend/API connection and try again.</div>
          {error && <div className="text-sm text-red-400">{error}</div>}
        </div>
      );
    }
    return (
      <div className="flex h-screen w-full items-center justify-center">
        <div className="h-10 w-10 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    );
  }

  return <>{children}</>;
}

export default AuthRedirect;
