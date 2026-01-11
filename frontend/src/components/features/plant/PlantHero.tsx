'use client';

import { Badge } from '@/components/ui/badge';
import { ArrowLeft } from 'lucide-react';

interface PlantHeroProps {
  name: string;
  species?: string;
  imageUrl?: string;
  healthPercent?: number; // e.g., 85
  onBack?: () => void;
  darkMode?: boolean;
}

export const PlantHero: React.FC<PlantHeroProps> = ({ name, species, imageUrl, healthPercent, onBack, darkMode }) => {
  return (
    <div className="relative w-full h-56 lg:h-72 xl:h-80 rounded-2xl overflow-hidden">
      {/* Image */}
      {imageUrl ? (
        <img src={imageUrl} alt={name} className="absolute inset-0 w-full h-full object-cover" />
      ) : (
        <div className="absolute inset-0 w-full h-full bg-neutral-200 dark:bg-neutral-800" />
      )}

      {/* Overlay gradient */}
      <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/30 to-transparent" />

      {/* Back button */}
      {onBack && (
        <button
          onClick={onBack}
          className="absolute top-4 left-4 z-40 w-10 h-10 rounded-full bg-white/80 dark:bg-neutral-800/80 backdrop-blur-sm shadow-lg flex items-center justify-center hover:bg-white dark:hover:bg-neutral-800 transition-colors"
          aria-label="Go back to collection"
        >
          <ArrowLeft className="w-5 h-5 text-neutral-900 dark:text-neutral-50" />
        </button>
      )}

      {/* Health badge */}
      {typeof healthPercent === 'number' && (
        <div className="absolute top-3 right-3">
          <Badge className="rounded-full bg-lime-500 text-black px-3 py-1 font-semibold">
            {healthPercent}% Health
          </Badge>
        </div>
      )}

      {/* Text content */}
      <div className="absolute bottom-3 left-3">
        <h2 className="text-white text-lg lg:text-xl font-semibold">{name}</h2>
        {species && (
          <p className="text-white/80 text-sm italic">{species}</p>
        )}
      </div>
    </div>
  );
}
