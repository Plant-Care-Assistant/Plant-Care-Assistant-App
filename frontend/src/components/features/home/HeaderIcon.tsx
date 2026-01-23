import Image from 'next/image';

interface HeaderIconProps {
  darkMode: boolean;
}

export function HeaderIcon({ darkMode }: HeaderIconProps) {
  return (
    <div className={`p-1 rounded-2xl border-2 ${darkMode ? 'border-neutral-700 bg-neutral-900' : 'border-neutral-200 bg-neutral-50'}`}>
      <div className="relative w-16 h-16">
        <Image
          src="/ic-8.png"
          alt="Profile"
          fill
          className="rounded-lg object-cover"
        />
      </div>
    </div>
  );
}
