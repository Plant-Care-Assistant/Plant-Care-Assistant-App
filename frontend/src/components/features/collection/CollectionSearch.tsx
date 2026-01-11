'use client';

import { Search } from 'lucide-react';

interface CollectionSearchProps {
  value: string;
  onChange: (value: string) => void;
  darkMode: boolean;
}

export const CollectionSearch: React.FC<CollectionSearchProps> = ({
  value,
  onChange,
  darkMode
}) => {
  return (
    <div className="mb-4">
      <div className={`relative rounded-xl shadow-sm ${
        darkMode 
          ? 'bg-neutral-800' 
          : 'bg-white'
      }`}>
        <div className="flex items-center px-4 py-3">
          <Search size={20} className={darkMode ? 'text-neutral-400' : 'text-neutral-500'} />
          <input
            type="text"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder="Search plants..."
            className={`ml-3 flex-1 bg-transparent outline-none text-sm ${
              darkMode 
                ? 'text-white placeholder:text-neutral-500' 
                : 'text-gray-900 placeholder:text-neutral-400'
            }`}
          />
        </div>
      </div>
    </div>
  );
};
