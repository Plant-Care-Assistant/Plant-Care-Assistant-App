"use client";

import Image from "next/image";
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
        "min-h-screen bg-neutral-50 text-neutral-900 dark:bg-neutral-950 dark:text-neutral-50",
        className,
      )}
    >
      {/* Desktop: side-by-side layout */}
      <div className="flex min-h-screen">
        {/* Form side */}
        <div className="relative z-10 flex w-full items-center justify-center px-4 py-8 lg:w-1/2">
          <div className="w-full max-w-sm">{children}</div>
        </div>

        {/* Web illustration — fixed right side, desktop only */}
        <div className="hidden lg:block lg:w-1/2">
          <div className="fixed right-0 top-0 h-screen w-[60vw] overflow-hidden">
            <img
              src="/web.png"
              alt=""
              className="h-full w-auto max-w-none"
              aria-hidden="true"
            />
          </div>
        </div>
      </div>

      {/* Mobile illustration — fixed to bottom, mobile/tablet only */}
      <div className="pointer-events-none fixed inset-x-0 bottom-0 z-0 lg:hidden">
        <Image
          src="/mobile.png"
          alt=""
          width={800}
          height={600}
          className="block w-full"
          priority
          aria-hidden="true"
        />
      </div>
    </div>
  );
}

export default AuthLayout;
