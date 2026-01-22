'use client';

import { Zap, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';

interface ScanHeaderProps {
  xpReward: number;
  darkMode: boolean;
}

export const ScanHeader: React.FC<ScanHeaderProps> = ({ xpReward, darkMode }) => {
  const [showInfo, setShowInfo] = useState(false);

  return (
    <>
      <div className="relative mb-4 lg:mb-6 flex items-center justify-center lg:justify-between lg:w-full">
        {/* XP Notification Badge - Floating design */}
        <motion.div 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="inline-flex items-center justify-center"
        >
          <div className={`inline-flex items-center gap-2 px-4 py-2.5 rounded-full shadow-lg backdrop-blur-sm border ${
            darkMode 
              ? 'bg-neutral-800/90 border-neutral-700/50 text-white' 
              : 'bg-white/90 border-neutral-200/50 text-gray-900'
          }`}>
            <motion.div
              animate={{ 
                scale: [1, 1.2, 1],
              }}
              transition={{ 
                duration: 2,
                repeat: Infinity,
                repeatDelay: 3
              }}
            >
              <Zap 
                size={18} 
                className="text-yellow-400" 
                fill="currentColor" 
                strokeWidth={2}
              />
            </motion.div>
            <span className="font-semibold text-sm tracking-tight">
              +{xpReward} XP per scan
            </span>
          </div>
        </motion.div>

        {/* Info button - positioned absolutely in top right of container */}
        <button
          onClick={() => setShowInfo(true)}
          className={`absolute right-0 top-1/2 -translate-y-1/2 lg:relative lg:top-0 lg:translate-y-0 w-9 h-9 rounded-full flex items-center justify-center transition-all ${
            darkMode 
              ? 'bg-neutral-800/60 text-neutral-400 hover:bg-neutral-700 hover:text-neutral-300 border border-neutral-700/50' 
              : 'bg-neutral-100/80 text-neutral-600 hover:bg-neutral-200 hover:text-neutral-800 border border-neutral-200/50'
          } hover:scale-105 active:scale-95`}
          title="Learn more about XP rewards"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 16v-4M12 8h.01" />
          </svg>
        </button>
      </div>

      {/* Info Modal */}
      <AnimatePresence>
        {showInfo && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowInfo(false)}
              className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
            />
            
            {/* Modal Content */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-full max-w-md mx-4"
            >
              <div className={`rounded-2xl shadow-2xl border ${
                darkMode
                  ? 'bg-neutral-800 border-neutral-700'
                  : 'bg-white border-neutral-200'
              } p-6`}>
                {/* Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Zap size={24} className="text-yellow-400" fill="currentColor" />
                    <h3 className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                      XP Rewards
                    </h3>
                  </div>
                  <button
                    onClick={() => setShowInfo(false)}
                    className={`p-1 rounded-lg transition-colors ${
                      darkMode
                        ? 'hover:bg-neutral-700 text-neutral-400'
                        : 'hover:bg-neutral-100 text-neutral-600'
                    }`}
                  >
                    <X size={20} />
                  </button>
                </div>

                {/* Content */}
                <div className={`space-y-4 ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
                  <div>
                    <h4 className={`font-semibold mb-1 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                      How it works
                    </h4>
                    <p className="text-sm">
                      Earn <span className="font-semibold text-yellow-400">+{xpReward} XP</span> every time you scan a plant! 
                      The more plants you identify, the more XP you collect to level up your plant care journey.
                    </p>
                  </div>

                  <div>
                    <h4 className={`font-semibold mb-1 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                      Level up benefits
                    </h4>
                    <ul className="text-sm space-y-1 ml-4">
                      <li>• Unlock advanced plant care tips</li>
                      <li>• Access exclusive plant guides</li>
                      <li>• Earn achievement badges</li>
                      <li>• Track your plant expertise</li>
                    </ul>
                  </div>

                  <div className={`p-3 rounded-lg ${
                    darkMode ? 'bg-neutral-700/50' : 'bg-neutral-50'
                  }`}>
                    <p className="text-xs">
                      💡 <span className="font-medium">Pro tip:</span> Clear, well-lit photos help identify plants 
                      accurately and ensure you get your XP reward!
                    </p>
                  </div>
                </div>

                {/* Close Button */}
                <button
                  onClick={() => setShowInfo(false)}
                  className="mt-6 w-full py-2.5 px-4 rounded-xl font-semibold transition-colors bg-secondary text-white hover:bg-secondary/90"
                >
                  Got it!
                </button>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
};
