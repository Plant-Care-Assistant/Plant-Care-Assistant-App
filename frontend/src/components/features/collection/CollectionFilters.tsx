'use client';

import { motion } from 'framer-motion';

export type FilterOption = 'all' | 'healthy' | 'need-care' | 'recent';

interface CollectionFiltersProps {
  activeFilter: FilterOption;
  onFilterChange: (filter: FilterOption) => void;
  darkMode: boolean;
}

const filters: { value: FilterOption; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'healthy', label: 'Healthy' },
  { value: 'need-care', label: 'Need Care' },
  { value: 'recent', label: 'Recent' },
];

export const CollectionFilters: React.FC<CollectionFiltersProps> = ({
  activeFilter,
  onFilterChange,
  darkMode
}) => {
  return (
    <div className="mb-6">
      <div className="flex items-center gap-2">
        {filters.map((filter) => (
          <button
            key={filter.value}
            onClick={() => onFilterChange(filter.value)}
            className={`relative flex-1 px-4 py-2.5 rounded-xl text-sm font-medium transition-all ${
              activeFilter === filter.value
                ? 'text-white'
                : darkMode
                ? 'text-neutral-400 bg-neutral-800 hover:bg-neutral-700 hover:text-neutral-300'
                : 'text-neutral-600 bg-white hover:bg-neutral-50 hover:text-neutral-800'
            }`}
          >
            {activeFilter === filter.value && (
              <motion.div
                layoutId="activeFilter"
                className="absolute inset-0 rounded-xl bg-secondary shadow-md"
                transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
              />
            )}
            <span className="relative z-10">{filter.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
};
