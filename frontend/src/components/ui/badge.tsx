import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils/cn";

/**
 * Badge variants for small status labels.
 */
export const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide",
  {
    variants: {
      variant: {
        primary: "border-transparent bg-primary/15 text-primary",
        secondary: "border-transparent bg-secondary/15 text-secondary",
        neutral:
          "border-transparent bg-neutral-100 text-neutral-800 dark:bg-neutral-800 dark:text-neutral-50",
        outline: "border-neutral-300 text-neutral-800 dark:border-neutral-700 dark:text-neutral-50",
      },
    },
    defaultVariants: {
      variant: "primary",
    },
  },
);

/**
 * Props for the Badge component.
 * @property variant Visual style variant.
 */
export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

/**
 * Small badge component for statuses or labels.
 */
export const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant, ...props }, ref) => (
    <span
      ref={ref}
      className={cn(badgeVariants({ variant }), className)}
      {...props}
    />
  ),
);

Badge.displayName = "Badge";

export default Badge;
