"use client";

import { motion } from "framer-motion";
import type { NavScreen } from "./navItems";
import { navItems } from "./navItems";

export interface BottomNavProps {
  currentScreen: NavScreen;
  onNavigate: (screen: NavScreen) => void;
  darkMode: boolean;
}

/**
 * Mobile-first sticky bottom navigation bar.
 */
export function BottomNav({ currentScreen, onNavigate, darkMode }: BottomNavProps) {
  return (
    <nav
      className={`fixed bottom-0 left-0 right-0 lg:hidden z-50 ${
        darkMode ? "bg-neutral-800" : "bg-white"
      } border-t ${darkMode ? "border-neutral-700" : "border-neutral-200"}`}
      style={{
        boxShadow: darkMode
          ? "0 -10px 15px -3px rgba(0, 0, 0, 0.5)"
          : "0 -10px 15px -3px rgba(0, 0, 0, 0.1)",
      }}
    >
      <div className="grid grid-cols-4 gap-1 p-2">
        {navItems.map((item) => {
          const isActive = currentScreen === item.screen;
          const Icon = item.icon;

          return (
            <motion.button
              key={item.screen}
              onClick={() => onNavigate(item.screen)}
              className={`flex flex-col items-center justify-center gap-1 py-3 px-2 rounded-2xl transition-all ${
                isActive
                  ? "bg-secondary text-white"
                  : darkMode
                    ? "text-neutral-400 hover:text-neutral-300"
                    : "text-neutral-600 hover:text-neutral-700"
              }`}
              whileTap={{ scale: 0.95 }}
            >
              <Icon size={24} />
              <span className="text-xs font-medium">{item.label}</span>
            </motion.button>
          );
        })}
      </div>
    </nav>
  );
}

export default BottomNav;
