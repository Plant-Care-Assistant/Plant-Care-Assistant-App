"use client";

import App from "@/components/App";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/providers";

export default function HomePage() {
  const { isAuthenticated, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading) {
      if (isAuthenticated) {
        router.replace("/dashboard");
      } else {
        router.replace("/auth/login");
      }
    }
  }, [isAuthenticated, isLoading, router]);

  return null;
}
