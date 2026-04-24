"use client";

import { cn } from "@/lib/utils/cn";
import { BottomNav } from "./BottomNav";
import { Header, type HeaderProps } from "./Header";
import { Sidebar } from "./Sidebar";
import { SCREEN_ROUTES, screenFromPathname, type NavScreen } from "./navItems";
import { useRouter, usePathname } from "next/navigation";

export interface LayoutProps {
  children: React.ReactNode;
  header?: HeaderProps | null;
  showBottomNav?: boolean;
  showSidebar?: boolean;
  onToggleDarkMode?: () => void;
  className?: string;
}

/** App shell providing consistent header, sidebar, bottom nav, and dark-mode chrome. */
export function Layout({
  children,
  header,
  showBottomNav = true,
  showSidebar = true,
  onToggleDarkMode,
  className,
}: LayoutProps) {
  const router = useRouter();
  const pathname = usePathname();
  const currentScreen = screenFromPathname(pathname);

  const handleNavigate = (screen: NavScreen) => {
    router.push(SCREEN_ROUTES[screen] ?? '/');
  };

  return (
    <div className={cn(
      "min-h-screen transition-colors duration-300",
      "bg-neutral-50 text-gray-900 dark:bg-neutral-900 dark:text-white",
      className,
    )}>
      {showSidebar && (
        <Sidebar
          currentScreen={currentScreen}
          onNavigate={handleNavigate}
          onToggleDarkMode={onToggleDarkMode}
          showThemeToggle={!!onToggleDarkMode}
        />
      )}

      <main className={showSidebar ? "lg:pl-72" : ""}>
        {header ? <Header {...header} /> : null}
        {children}
      </main>

      {showBottomNav && (
        <BottomNav currentScreen={currentScreen} onNavigate={handleNavigate} />
      )}
    </div>
  );
}

export default Layout;
