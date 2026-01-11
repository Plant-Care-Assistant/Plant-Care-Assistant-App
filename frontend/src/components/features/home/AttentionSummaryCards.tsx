import { Droplet, Sun } from 'lucide-react';

interface AttentionSummaryCardsProps {
  needWater: number;
  lowLight: number;
  darkMode: boolean;
}

export function AttentionSummaryCards({ needWater, lowLight, darkMode }: AttentionSummaryCardsProps) {
  return (
    <div className="grid grid-cols-2 gap-4">
      {/* Need Water */}
      <div className={`p-4 rounded-2xl ${darkMode ? 'bg-neutral-800' : 'bg-white'} shadow-md flex items-center gap-3`}>
        <div className="w-10 h-10 rounded-xl bg-primary/80 flex items-center justify-center shrink-0">
          <Droplet className="w-5 h-5 text-white" />
        </div>
        <div>
          <p className={`text-2xl font-semibold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>{needWater}</p>
          <p className={`text-xs ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>need water</p>
        </div>
      </div>

      {/* Low Light */}
      <div className={`p-4 rounded-2xl ${darkMode ? 'bg-neutral-800' : 'bg-white'} shadow-md flex items-center gap-3`}>
        <div className="w-10 h-10 rounded-xl bg-nature flex items-center justify-center shrink-0">
          <Sun className="w-5 h-5 text-white" />
        </div>
        <div>
          <p className={`text-2xl font-semibold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>{lowLight}</p>
          <p className={`text-xs ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>low light</p>
        </div>
      </div>
    </div>
  );
}
