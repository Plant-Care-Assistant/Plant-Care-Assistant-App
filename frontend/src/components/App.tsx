'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BottomNav } from '@/components/layout/BottomNav';
import { Sidebar } from '@/components/layout/Sidebar';
import { HomeScreen } from '@/components/screens/HomeScreen';
import { ScanScreen } from '@/components/screens/ScanScreen';
import { CollectionScreen } from '@/components/screens/CollectionScreen';
import { ProfileScreen } from '@/components/screens/ProfileScreen';
import type { NavScreen } from '@/components/layout/navItems';
import { MOCK_PLANTS, type Plant } from '@/lib/utils/plantFilters';

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<NavScreen>('home');
  const [darkMode, setDarkMode] = useState(false);
  const [plants, setPlants] = useState<Plant[]>(MOCK_PLANTS);

  // Apply dark mode class to document
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  const renderScreen = () => {
    switch (currentScreen) {
      case 'home':
        return <HomeScreen darkMode={darkMode} plants={plants} />;
      case 'scan':
        return <ScanScreen darkMode={darkMode} plants={plants} onPlantsChange={setPlants} />;
      case 'collection':
        return <CollectionScreen darkMode={darkMode} />;
      case 'profile':
        return <ProfileScreen darkMode={darkMode} onDarkModeToggle={toggleDarkMode} />;
      default:
        return <HomeScreen darkMode={darkMode} plants={plants} />;
    }
  };

  return (
    <div className={`min-h-screen transition-colors duration-300 ${
      darkMode 
        ? 'bg-neutral-900 text-white' 
        : 'bg-neutral-50 text-gray-900'
    }`}>
      {/* Desktop Sidebar Navigation */}
      <Sidebar 
        darkMode={darkMode}
        currentScreen={currentScreen}
        onNavigate={setCurrentScreen}
        onToggleDarkMode={toggleDarkMode}
        showThemeToggle={true}
      />

      {/* Main Content */}
      <main className={`lg:pl-72 transition-all ${
        darkMode ? 'bg-neutral-900' : 'bg-neutral-50'
      }`}>
        <AnimatePresence mode="wait">
          <motion.div
            key={currentScreen}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
          >
            {renderScreen()}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Bottom Navigation - Only visible on mobile/tablet */}
      <BottomNav 
        currentScreen={currentScreen} 
        onNavigate={setCurrentScreen}
        darkMode={darkMode}
      />
    </div>
  );
}
