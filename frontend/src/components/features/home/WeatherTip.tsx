import { Cloud } from 'lucide-react';

interface WeatherTipProps {
  title: string;
  description: string;
  darkMode: boolean;
}

export function WeatherTip({ title, description, darkMode }: WeatherTipProps) {
  return (
    <div className={`p-5 rounded-3xl ${darkMode ? 'bg-neutral-800' : 'bg-accent'} shadow-md mb-6 flex items-start gap-3`}>
      <Cloud className="w-6 h-6 text-white flex-shrink-0 mt-1" />
      <div>
        <h3 className="text-sm font-bold text-white mb-1">
          {title}
        </h3>
        <p className="text-xs text-white/80">
          {description}
        </p>
      </div>
    </div>
  );
}
