"use client";

import { motion } from "framer-motion";
import type { NavScreen } from "./navItems";
import { navItems } from "./navItems";

export interface BottomNavProps {
  currentScreen: NavScreen;
  onNavigate: (screen: NavScreen) => void;
}

/** Mobile-first sticky bottom navigation bar. */
export function BottomNav({ currentScreen, onNavigate }: BottomNavProps) {
  return (
    <nav className="fixed bottom-0 left-0 right-0 lg:hidden z-50 bg-white border-t border-neutral-200 dark:bg-neutral-800 dark:border-neutral-700 shadow-[0_-10px_15px_-3px_rgba(0,0,0,0.1)] dark:shadow-[0_-10px_15px_-3px_rgba(0,0,0,0.5)]">
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
                  ? 'bg-secondary text-white'
                  : 'text-neutral-600 hover:text-neutral-700 dark:text-neutral-400 dark:hover:text-neutral-300'
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
