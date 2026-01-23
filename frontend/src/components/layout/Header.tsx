"use client";

import { ArrowLeft } from "lucide-react";
import { useRouter } from "next/navigation";
import { cn } from "@/lib/utils/cn";
import { Button } from "@/components/ui";

/**
 * Props for the Header component.
 * @property title Main heading text.
 * @property subtitle Optional supporting text.
 * @property rightSlot Optional custom content on the right side.
 * @property onBack Optional custom back handler.
 * @property showBack Whether to render a back button (defaults to false).
 */
export interface HeaderProps {
  title: string;
  subtitle?: string;
  rightSlot?: React.ReactNode;
  onBack?: () => void;
  showBack?: boolean;
  className?: string;
}

/**
 * Top application header with optional back button and action slot.
 */
export function Header({ title, subtitle, rightSlot, onBack, showBack = false, className }: HeaderProps) {
  const router = useRouter();
  const handleBack = () => {
    if (onBack) return onBack();
    router.back();
  };

  return (
    <header
      className={cn(
        "sticky top-0 z-40 flex items-center justify-between gap-3 border-b",
        "bg-white/90 backdrop-blur supports-[backdrop-filter]:bg-white/80 border-neutral-200",
        "dark:bg-neutral-900/90 dark:supports-[backdrop-filter]:bg-neutral-900/80 dark:border-neutral-800",
        "px-4 py-3",
        className,
      )}
    >
      <div className="flex items-center gap-3">
        {showBack && (
          <Button
            variant="ghost"
            size="icon"
            aria-label="Go back"
            onClick={handleBack}
            className="shrink-0"
          >
            <ArrowLeft className="h-5 w-5" aria-hidden="true" />
          </Button>
        )}
        <div className="flex flex-col">
          <span className="text-lg font-semibold leading-tight text-neutral-900 dark:text-neutral-50">{title}</span>
          {subtitle && (
            <span className="text-sm text-neutral-600 dark:text-neutral-400">{subtitle}</span>
          )}
        </div>
      </div>
      {rightSlot ? <div className="flex items-center gap-2">{rightSlot}</div> : null}
    </header>
  );
}

export default Header;
