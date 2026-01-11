'use client';

import { Droplet, CalendarDays, TrendingUp } from 'lucide-react';

type StatType = 'watered' | 'cycle' | 'health';

interface StatCardProps {
  type: StatType;
  value: string | number; // e.g., '2 days ago' | 5 | 85
  label?: string; // optional label override
  darkMode?: boolean;
}

const colorForType = (type: StatType) => {
  switch (type) {
    case 'watered':
      return 'bg-accent'; // Light blue
    case 'cycle':
      return 'bg-primary'; // Blue
    case 'health':
      return 'bg-secondary'; // Green
  }
};

const iconForType = (type: StatType) => {
  switch (type) {
    case 'watered':
      return <Droplet size={24} />;
    case 'cycle':
      return <CalendarDays size={24} />;
    case 'health':
      return <TrendingUp size={24} />;
  }
};

const defaultLabel = (type: StatType) => {
  switch (type) {
    case 'watered':
      return 'watered';
    case 'cycle':
      return 'day cycle';
    case 'health':
      return 'healthy';
  }
};

export const StatCard: React.FC<StatCardProps> = ({ type, value, label, darkMode = false }) => {
  const bgColor = colorForType(type);
  
  return (
    <div className={`flex-1 rounded-3xl shadow-lg p-3 sm:p-6 flex flex-col items-center text-center ${bgColor}`}>
      <span className="text-white mt-2 sm:mt-4">{iconForType(type)}</span>
      <span className="text-2xl sm:text-3xl font-bold text-white mt-3 sm:mt-6 mb-1">
        {typeof value === 'number' ? value : value}
        {type === 'health' && <span className="text-lg sm:text-2xl">%</span>}
      </span>
      <span className="text-xs sm:text-sm text-white">{label || defaultLabel(type)}</span>
    </div>
  );
};
