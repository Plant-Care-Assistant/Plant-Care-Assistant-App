'use client';

import { Droplet, Wind, Sprout, Scissors, RotateCw, Eye, MoreHorizontal, ChevronDown } from 'lucide-react';
import { ActionButton } from './ActionButton';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { CareType } from '@/types';

interface PlantActionsProps {
  onWaterNow?: () => void;
  onLogCare?: (type: Exclude<CareType, 'water'>) => void;
  darkMode?: boolean;
}

const NON_WATER_OPTIONS: Array<{
  type: Exclude<CareType, 'water'>;
  label: string;
  Icon: typeof Wind;
}> = [
  { type: 'mist', label: 'Mist', Icon: Wind },
  { type: 'fertilize', label: 'Fertilize', Icon: Sprout },
  { type: 'prune', label: 'Prune', Icon: Scissors },
  { type: 'rotate', label: 'Rotate pot', Icon: RotateCw },
  { type: 'inspect', label: 'Inspect', Icon: Eye },
  { type: 'other', label: 'Other care', Icon: MoreHorizontal },
];

export const PlantActions: React.FC<PlantActionsProps> = ({ onWaterNow, onLogCare, darkMode = false }) => {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      <ActionButton
        icon={<Droplet size={24} fill="currentColor" />}
        label="Water Now"
        onClick={onWaterNow}
        variant="water"
        darkMode={darkMode}
      />

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            className={`
              w-full flex flex-col sm:flex-row items-center justify-center gap-2 sm:gap-3
              px-4 sm:px-6 py-5 sm:py-4 rounded-3xl
              font-semibold text-sm sm:text-base
              transition-all duration-200
              ${darkMode
                ? 'border-2 border-dashed border-neutral-700 text-neutral-300 hover:border-nature hover:bg-neutral-800/30'
                : 'border-2 border-dashed border-neutral-300 text-neutral-700 hover:border-nature hover:bg-neutral-50'}
              text-nature
            `}
          >
            <span className="flex items-center justify-center text-2xl sm:text-xl">
              <Sprout size={24} fill="currentColor" />
            </span>
            <span className="flex items-center gap-1">
              Other care <ChevronDown size={16} />
            </span>
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="min-w-[180px]">
          {NON_WATER_OPTIONS.map(({ type, label, Icon }) => (
            <DropdownMenuItem
              key={type}
              onSelect={() => onLogCare?.(type)}
              className="gap-2"
            >
              <Icon size={16} />
              {label}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
};
