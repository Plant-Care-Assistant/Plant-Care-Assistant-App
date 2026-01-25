"use client";

import Image from "next/image";
import { Moon, Sun, LogOut } from "lucide-react";
import { navItems, type NavScreen } from "./navItems";
import { useAuth } from "@/providers";

/**
 * Props for the Sidebar component.
 */
export interface SidebarProps {
  darkMode?: boolean;
  currentScreen?: NavScreen;
  onNavigate?: (screen: NavScreen) => void;
  onToggleDarkMode?: () => void;
  showThemeToggle?: boolean;
}

/**
 * Desktop sidebar navigation component used across different layouts.
 * Contains branding, navigation items, and optional theme toggle.
 */
export function Sidebar({ 
  darkMode = false,
  currentScreen = 'home',
  onNavigate,
  onToggleDarkMode,
  showThemeToggle = true
}: SidebarProps) {
  const { logout } = useAuth();
  return (
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
            onClick={() => onNavigate?.(item.screen)}
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
      {showThemeToggle && onToggleDarkMode && (
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
      {/* Log out Button */}
      <div className="border-t p-4">
        <button
          onClick={logout}
          className={`w-full px-4 py-3 rounded-lg transition-colors flex items-center justify-center gap-2 font-medium text-red-600 hover:bg-red-100 dark:hover:bg-neutral-800`}
        >
          <LogOut size={18} />
          Log out
        </button>
      </div>
    </nav>
  );
}
