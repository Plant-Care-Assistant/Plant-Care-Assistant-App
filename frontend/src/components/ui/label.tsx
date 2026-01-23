import * as React from "react";
import { cn } from "@/lib/utils/cn";

/**
 * Props for the Label component.
 */
export interface LabelProps extends React.LabelHTMLAttributes<HTMLLabelElement> {}

/**
 * Accessible label component for form controls.
 */
export const Label = React.forwardRef<HTMLLabelElement, LabelProps>(
  ({ className, ...props }, ref) => (
    <label
      ref={ref}
      className={cn(
        "text-sm font-medium text-neutral-800 dark:text-neutral-200",
        className,
      )}
      {...props}
    />
  ),
);

Label.displayName = "Label";

export default Label;
