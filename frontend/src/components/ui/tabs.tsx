"use client";

import * as React from "react";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import { cn } from "@/lib/utils/cn";

/**
 * Props for the Tabs component.
 */
export interface TabsProps extends TabsPrimitive.TabsProps {}

/**
 * Root tabs container.
 */
export const Tabs = ({ className, ...props }: TabsProps) => (
  <TabsPrimitive.Root className={cn("flex flex-col gap-3", className)} {...props} />
);

export interface TabsListProps extends TabsPrimitive.TabsListProps {}
export const TabsList = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.List>,
  TabsListProps
>(({ className, ...props }, ref) => (
  <TabsPrimitive.List
    ref={ref}
    className={cn(
      "inline-flex w-full items-center justify-start gap-2 rounded-xl bg-neutral-100 p-1 dark:bg-neutral-800",
      className,
    )}
    {...props}
  />
));
TabsList.displayName = "TabsList";

export interface TabsTriggerProps extends TabsPrimitive.TabsTriggerProps {}
export const TabsTrigger = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Trigger>,
  TabsTriggerProps
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Trigger
    ref={ref}
    className={cn(
      "flex-1 whitespace-nowrap rounded-lg px-3 py-2 text-sm font-medium text-neutral-700 transition data-[state=active]:bg-white data-[state=active]:text-neutral-900 data-[state=active]:shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 dark:text-neutral-300 dark:data-[state=active]:bg-neutral-900 dark:data-[state=active]:text-white",
      className,
    )}
    {...props}
  />
));
TabsTrigger.displayName = "TabsTrigger";

export interface TabsContentProps extends TabsPrimitive.TabsContentProps {}
export const TabsContent = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Content>,
  TabsContentProps
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Content
    ref={ref}
    className={cn("focus-visible:outline-none", className)}
    {...props}
  />
));
TabsContent.displayName = "TabsContent";

export default Tabs;
