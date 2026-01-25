"use client";

import { useEffect } from "react";
import { useRouter, usePathname } from "next/navigation";
import { useAuth } from "@/providers";

interface ProtectedRouteProps {
  children: React.ReactNode;
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { isAuthenticated, isLoading, token } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    // If not loading and not authenticated, redirect to login
    if (!isLoading && !isAuthenticated) {
      // Save intended path for redirect after login
      if (typeof window !== "undefined") {
        sessionStorage.setItem("redirectAfterLogin", pathname || "/");
      }
      router.replace("/auth/login");
    }
  }, [isAuthenticated, isLoading, pathname, router]);

  // Optionally: handle token expiration (if you parse JWT expiry)
  // useEffect(() => {
  //   if (token) {
  //     const payload = JSON.parse(atob(token.split(".")[1]));
  //     if (payload.exp && Date.now() / 1000 > payload.exp) {
  //       router.replace("/auth/login");
  //     }
  //   }
  // }, [token, router]);

  if (isLoading) {
    return (
      <div className="flex h-screen w-full items-center justify-center">
        <div className="h-10 w-10 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    );
  }

  if (!isAuthenticated) {
    // Optionally: render nothing while redirecting
    return null;
  }

  return <>{children}</>;
}

export default ProtectedRoute;
