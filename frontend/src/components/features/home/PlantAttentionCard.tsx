import { Droplet } from 'lucide-react';

interface PlantAttentionCardProps {
  plantName: string;
  hoursOverdue: number;
  onWaterClick: () => void;
  darkMode: boolean;
}

export function PlantAttentionCard({ plantName, hoursOverdue, onWaterClick, darkMode }: PlantAttentionCardProps) {
  return (
    <div className={`p-4 rounded-3xl ${darkMode ? 'bg-neutral-800' : 'bg-white'} shadow-md mb-3 flex items-center justify-between`}>
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-primary-light flex items-center justify-center">
          <Droplet className="w-5 h-5 text-white" />
        </div>
        <div>
          <h4 className={`font-semibold text-sm ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            {plantName}
          </h4>
          <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            {hoursOverdue}h overdue
          </p>
        </div>
      </div>
      <button
        onClick={onWaterClick}
        className="px-4 py-2 bg-primary-dark text-white rounded-xl text-xs font-bold hover:bg-primary transition-colors flex items-center gap-2 flex-shrink-0"
      >
        Water
      </button>
    </div>
  );
}
