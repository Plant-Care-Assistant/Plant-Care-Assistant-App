"use client";

import { cn } from "@/lib/utils/cn";
import { BottomNav, type BottomNavProps } from "./BottomNav";
import { Header, type HeaderProps } from "./Header";
import { Sidebar } from "./Sidebar";
import { useRouter, usePathname } from "next/navigation";

/**
 * Props for the Layout wrapper.
 * @property children Page content.
 * @property header Optional header configuration; omit to hide header.
 * @property showBottomNav Toggle bottom navigation (defaults to true on mobile use-cases).
 * @property showSidebar Toggle desktop sidebar (defaults to true).
 * @property darkMode Dark mode state.
 * @property onToggleDarkMode Dark mode toggle handler.
 */
export interface LayoutProps {
  children: React.ReactNode;
  header?: HeaderProps | null;
  showBottomNav?: boolean;
  showSidebar?: boolean;
  darkMode?: boolean;
  onToggleDarkMode?: () => void;
  className?: string;
}

/**
 * App shell wrapper providing consistent header, main content spacing, sidebar, and bottom nav.
 */
export function Layout({ 
  children, 
  header, 
  showBottomNav = true, 
  showSidebar = true,
  darkMode = false,
  onToggleDarkMode,
  className 
}: LayoutProps) {
  const router = useRouter();
  const pathname = usePathname();

  const getCurrentScreen = () => {
    if (pathname === '/') return 'home';
    if (pathname?.startsWith('/scan')) return 'scan';
    if (pathname?.startsWith('/plant')) return 'collection';
    if (pathname?.startsWith('/collection')) return 'collection';
    if (pathname?.startsWith('/profile')) return 'profile';
    return 'home';
  };

  const currentScreen = getCurrentScreen();

  const handleNavigate = (screen: string) => {
    const routes = {
      home: '/',
      scan: '/scan',
      collection: '/collection',
      profile: '/profile'
    };
    router.push(routes[screen as keyof typeof routes] || '/');
  };

  return (
    <div className={cn("min-h-screen transition-colors duration-300", 
      darkMode ? 'bg-neutral-900 text-white' : 'bg-neutral-50 text-gray-900',
      className
    )}>
      {/* Desktop Sidebar Navigation */}
      {showSidebar && (
        <Sidebar 
          darkMode={darkMode}
          currentScreen={currentScreen}
          onNavigate={handleNavigate}
          onToggleDarkMode={onToggleDarkMode}
          showThemeToggle={!!onToggleDarkMode}
        />
      )}

      {/* Main Content */}
      <main className={showSidebar ? "lg:pl-72" : ""}>
        {header ? <Header {...header} /> : null}
        {children}
      </main>

      {/* Bottom Navigation - Only visible on mobile/tablet */}
      {showBottomNav && (
        <BottomNav 
          currentScreen={currentScreen as any}
          onNavigate={(screen) => {
            const routes = {
              home: '/',
              scan: '/scan',
              collection: '/collection',
              profile: '/profile'
            };
            router.push(routes[screen] || '/');
          }}
          darkMode={darkMode}
        />
      )}
    </div>
  );
}

export default Layout;
