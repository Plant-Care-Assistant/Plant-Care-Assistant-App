'use client';

import { ThermometerSun, Sun } from 'lucide-react';

type EnvType = 'temperature' | 'light';

interface EnvInfoCardProps {
  type: EnvType;
  value: string;
  subtitle?: string;
  darkMode?: boolean;
}

const iconForType = (type: EnvType) => {
  switch (type) {
    case 'temperature':
      return <ThermometerSun className="text-[#64B5F6]" size={20} />;
    case 'light':
      return <Sun className="text-[#64B5F6]" size={20} />;
  }
};

const titleForType = (type: EnvType) => {
  switch (type) {
    case 'temperature':
      return 'Temperature';
    case 'light':
      return 'Light';
  }
};

export const EnvInfoCard: React.FC<EnvInfoCardProps> = ({ type, value, subtitle, darkMode = false }) => {
  return (
    <div className={`rounded-2xl shadow-sm p-4 ${darkMode ? 'bg-neutral-800' : 'bg-white'}`}>
      <div className="flex items-center gap-2 mb-2">
        <div className="w-9 h-9 rounded-xl bg-[#E3F2FD] flex items-center justify-center">
          {iconForType(type)}
        </div>
        <span className={`text-sm font-medium ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>{titleForType(type)}</span>
      </div>
      <div>
        <p className={`font-semibold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>{value}</p>
        {subtitle && (
          <p className={`text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>{subtitle}</p>
        )}
      </div>
    </div>
  );
};
