import { cn } from "@/lib/utils/cn";

/**
 * Props for the Skeleton component.
 */
export interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {}

/**
 * Animated placeholder for loading states.
 */
export function Skeleton({ className, ...props }: SkeletonProps) {
  return (
    <div
      role="status"
      aria-live="polite"
      className={cn(
        "animate-pulse rounded-md bg-neutral-200 dark:bg-neutral-800",
        className,
      )}
      {...props}
    />
  );
}

export default Skeleton;
