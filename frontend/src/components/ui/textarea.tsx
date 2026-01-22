import * as React from "react";
import { cn } from "@/lib/utils/cn";

/**
 * Props for the Textarea component.
 * @property isInvalid Marks the textarea as invalid for accessibility.
 */
export interface TextareaProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  isInvalid?: boolean;
}

/**
 * Multiline text input with consistent styling and focus states.
 */
export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, isInvalid, ...props }, ref) => (
    <textarea
      ref={ref}
      aria-invalid={isInvalid || props["aria-invalid"]}
      className={cn(
        "flex min-h-[120px] w-full rounded-md border border-neutral-300 bg-white px-3 py-2 text-sm text-neutral-900 shadow-sm transition focus-visible:border-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/70 dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-50",
        "placeholder:text-neutral-500 dark:placeholder:text-neutral-400 disabled:cursor-not-allowed disabled:opacity-60",
        isInvalid && "border-red-500 focus-visible:ring-red-400",
        className,
      )}
      {...props}
    />
  ),
);

Textarea.displayName = "Textarea";

export default Textarea;
