'use client';

import { Droplet, Zap } from 'lucide-react';
import { ActionButton } from './ActionButton';

interface PlantActionsProps {
  onWaterNow?: () => void;
  onGainXP?: () => void;
  darkMode?: boolean;
}

export const PlantActions: React.FC<PlantActionsProps> = ({ onWaterNow, onGainXP, darkMode = false }) => {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      <ActionButton
        icon={<Droplet size={24} fill="currentColor" />}
        label="Water Now"
        onClick={onWaterNow}
        variant="water"
        darkMode={darkMode}
      />
      <ActionButton
        icon={<Zap size={24} fill="currentColor" />}
        label="+10 XP"
        onClick={onGainXP}
        variant="xp"
        darkMode={darkMode}
      />
    </div>
  );
};
