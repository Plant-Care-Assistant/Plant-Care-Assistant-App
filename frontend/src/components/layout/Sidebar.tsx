"use client";

import Image from "next/image";
import { Moon, Sun, LogOut } from "lucide-react";
import { useAuth, useTheme } from "@/providers";
import { navItems, type NavScreen } from "./navItems";

export interface SidebarProps {
  currentScreen?: NavScreen;
  onNavigate?: (screen: NavScreen) => void;
  onToggleDarkMode?: () => void;
  showThemeToggle?: boolean;
}

/** Desktop sidebar with branding, nav, theme toggle, and logout. */
export function Sidebar({
  currentScreen = 'home',
  onNavigate,
  onToggleDarkMode,
  showThemeToggle = true,
}: SidebarProps) {
  const { logout } = useAuth();
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  return (
    <nav className="hidden lg:block fixed left-0 top-0 h-screen w-72 border-r transition-colors bg-white border-neutral-200 dark:bg-neutral-800 dark:border-neutral-700 shadow-[2px_0_15px_-3px_rgba(0,0,0,0.1)] dark:shadow-[2px_0_15px_-3px_rgba(0,0,0,0.5)]">
      <div className="border-b p-6 border-neutral-200 dark:border-neutral-700">
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

      <div className="p-4 space-y-2 flex-1">
        {navItems.map((item) => {
          const isActive = currentScreen === item.screen;
          return (
            <button
              key={item.screen}
              onClick={() => onNavigate?.(item.screen)}
              className={`w-full text-left px-4 py-3 rounded-lg transition-colors flex items-center gap-3 ${
                isActive
                  ? 'bg-secondary text-white'
                  : 'text-neutral-600 hover:bg-neutral-100 dark:text-neutral-300 dark:hover:bg-neutral-700'
              }`}
            >
              <item.icon size={20} />
              <span className="font-medium">{item.label}</span>
            </button>
          );
        })}
      </div>

      <div className="border-t p-4 space-y-2 border-neutral-200 dark:border-neutral-700">
        {showThemeToggle && onToggleDarkMode && (
          <button
            onClick={onToggleDarkMode}
            className="w-full px-4 py-3 rounded-lg transition-colors flex items-center justify-center gap-2 font-medium bg-neutral-200 text-primary hover:bg-neutral-300 dark:bg-neutral-700 dark:text-accent2 dark:hover:bg-neutral-600"
          >
            {isDark ? (
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
        )}
        <button
          onClick={logout}
          className="w-full px-4 py-3 rounded-lg transition-colors flex items-center justify-center gap-2 font-medium text-accent2 hover:bg-pink-50 dark:hover:bg-neutral-700"
        >
          <LogOut size={18} />
          Log out
        </button>
      </div>
    </nav>
  );
}
