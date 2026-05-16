import Image from 'next/image';
import { iconForLevel } from '@/lib/utils/levelIcon';

interface HeaderIconProps {
  level: number;
  darkMode: boolean;
}

export function HeaderIcon({ level, darkMode }: HeaderIconProps) {
  return (
    <div className={`p-1 rounded-2xl border-2 ${darkMode ? 'border-neutral-700 bg-neutral-900' : 'border-neutral-200 bg-neutral-50'}`}>
      <div className="relative w-16 h-16">
        <Image
          src={iconForLevel(level)}
          alt="Profile"
          fill
          className="rounded-lg object-cover"
        />
      </div>
    </div>
  );
}
