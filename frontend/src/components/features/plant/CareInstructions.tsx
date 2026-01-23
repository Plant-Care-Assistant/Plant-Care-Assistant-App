'use client';

import { CheckCircle2 } from 'lucide-react';

interface CareInstructionsProps {
  items: string[];
  darkMode?: boolean;
}

export const CareInstructions: React.FC<CareInstructionsProps> = ({ items, darkMode = false }) => {
  return (
    <div className={`rounded-2xl shadow-sm p-4 ${darkMode ? 'bg-neutral-800' : 'bg-white'}`}>
      <p className={`text-sm font-medium mb-3 ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>Care Instructions</p>
      <div className="space-y-2">
        {items.map((item, idx) => (
          <div key={idx} className="flex items-center gap-3">
            <span className="w-6 h-6 rounded-full bg-[#E6F4EA] flex items-center justify-center">
              <CheckCircle2 className="text-[#2E7D32]" size={18} />
            </span>
            <span className={`text-sm ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>{item}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
