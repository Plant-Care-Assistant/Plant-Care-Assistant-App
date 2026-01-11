"use client";

import { cn } from "@/lib/utils/cn";
import { BottomNav, type BottomNavProps } from "./BottomNav";
import { Header, type HeaderProps } from "./Header";
import { navItems } from "./navItems";
import Image from "next/image";
import { Moon, Sun } from "lucide-react";
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

  return (
    <div className={cn("min-h-screen transition-colors duration-300", 
      darkMode ? 'bg-neutral-900 text-white' : 'bg-neutral-50 text-gray-900',
      className
    )}>
      {/* Desktop Sidebar Navigation - Only visible on lg and above */}
      {showSidebar && (
        <nav className={`hidden lg:block fixed left-0 top-0 h-screen w-72 border-r transition-colors ${
          darkMode 
            ? 'bg-neutral-800 border-neutral-700' 
            : 'bg-white border-neutral-200'
        }`} style={{
          boxShadow: darkMode
            ? '2px 0 15px -3px rgba(0, 0, 0, 0.5)'
            : '2px 0 15px -3px rgba(0, 0, 0, 0.1)',
        }}>
          {/* Branding */}
          <div className="border-b p-6" style={{
            borderColor: darkMode ? '#374151' : '#e5e7eb',
          }}>
            <div className="flex items-center justify-start gap-3">
              <div className="w-16 h-16 rounded-3xl bg-secondary flex items-center justify-center p-2 flex-shrink-0">
                <Image 
                  src="/logo.png" 
                  alt="Plant Care Logo" 
                  width={56} 
                  height={56}
                  priority
                  className="w-full h-full object-contain"
                />
              </div>
              <div className="flex flex-col justify-center">
                <h1 className="font-bold text-xl">PlantCare</h1>
                <h2 className="font-bold text-xl -mt-2">Assistant</h2>
              </div>
            </div>
          </div>

          {/* Nav Items */}
          <div className="p-4 space-y-2 flex-1">
            {navItems.map((item) => (
              <button
                key={item.screen}
                onClick={() => {
                  const routes = {
                    home: '/',
                    scan: '/scan',
                    collection: '/collection',
                    profile: '/profile'
                  };
                  router.push(routes[item.screen] || '/');
                }}
                className={`w-full text-left px-4 py-3 rounded-lg transition-colors flex items-center gap-3 ${
                  currentScreen === item.screen
                    ? 'bg-secondary text-white'
                    : darkMode
                      ? 'text-neutral-300 hover:bg-neutral-700'
                      : 'text-neutral-600 hover:bg-neutral-100'
                }`}
              >
                <item.icon size={20} />
                <span className="font-medium">{item.label}</span>
              </button>
            ))}
          </div>

          {/* Theme Toggle */}
          {onToggleDarkMode && (
            <div className="border-t p-4" style={{
              borderColor: darkMode ? '#374151' : '#e5e7eb',
            }}>
              <button
                onClick={onToggleDarkMode}
                className={`w-full px-4 py-3 rounded-lg transition-colors flex items-center justify-center gap-2 font-medium ${
                  darkMode
                    ? 'bg-neutral-700 text-accent2 hover:bg-neutral-600'
                    : 'bg-neutral-200 text-primary hover:bg-neutral-300'
                }`}
              >
                {darkMode ? (
                  <>
                    <Sun size={18} />
                    Light Mode
                  </>
                ) : (
                  <>
                    <Moon size={18} />
                    Dark Mode
                  </>
                )}
              </button>
            </div>
          )}
        </nav>
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
