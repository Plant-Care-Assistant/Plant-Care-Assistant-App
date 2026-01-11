'use client';

import Image from 'next/image';
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BottomNav } from '@/components/layout/BottomNav';
import { HomeScreen } from '@/components/screens/HomeScreen';
import { ScanScreen } from '@/components/screens/ScanScreen';
import { CollectionScreen } from '@/components/screens/CollectionScreen';
import { ProfileScreen } from '@/components/screens/ProfileScreen';
import type { NavScreen } from '@/components/layout/navItems';
import { navItems } from '@/components/layout/navItems';
import { Moon, Sun } from 'lucide-react';
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
        return <ScanScreen darkMode={darkMode} />;
      case 'collection':
        return <CollectionScreen darkMode={darkMode} plants={plants} onPlantsChange={setPlants} />;
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
      {/* Desktop Sidebar Navigation - Only visible on lg and above */}
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
              onClick={() => setCurrentScreen(item.screen)}
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
        <div className="border-t p-4" style={{
          borderColor: darkMode ? '#374151' : '#e5e7eb',
        }}>
          <button
            onClick={toggleDarkMode}
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
      </nav>

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
