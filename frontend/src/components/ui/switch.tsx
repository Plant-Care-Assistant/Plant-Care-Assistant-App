"use client";

import * as React from "react";
import * as SwitchPrimitive from "@radix-ui/react-switch";
import { cn } from "@/lib/utils/cn";

/**
 * Props for the Switch component.
 */
export interface SwitchProps extends SwitchPrimitive.SwitchProps {}

/**
 * Toggle switch built on Radix.
 */
export const Switch = React.forwardRef<
  React.ElementRef<typeof SwitchPrimitive.Root>,
  SwitchProps
>(({ className, ...props }, ref) => (
  <SwitchPrimitive.Root
    ref={ref}
    className={cn(
      "peer inline-flex h-6 w-11 shrink-0 items-center rounded-full border border-transparent bg-neutral-300 transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60 focus-visible:ring-offset-2 data-[state=checked]:bg-primary disabled:cursor-not-allowed disabled:opacity-60 dark:bg-neutral-700",
      className,
    )}
    {...props}
  >
    <SwitchPrimitive.Thumb
      className={cn(
        "pointer-events-none block h-5 w-5 rounded-full bg-white shadow transition-transform data-[state=checked]:translate-x-5 data-[state=unchecked]:translate-x-0 dark:bg-neutral-100",
      )}
    />
  </SwitchPrimitive.Root>
));

Switch.displayName = "Switch";

export default Switch;
