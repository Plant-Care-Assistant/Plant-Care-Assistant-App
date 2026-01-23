'use client';

import { ReactNode } from 'react';

interface ActionButtonProps {
  icon: ReactNode;
  label: string;
  onClick?: () => void;
  variant?: 'water' | 'xp';
  darkMode?: boolean;
}

const waterButtonStyle = 'bg-accent text-white shadow-lg hover:opacity-90';

const xpOutlineStyle = {
  light: 'border-2 border-dashed border-neutral-300 text-neutral-700 hover:border-nature hover:bg-neutral-50',
  dark: 'border-2 border-dashed border-neutral-700 text-neutral-300 hover:border-nature hover:bg-neutral-800/30',
};

export const ActionButton: React.FC<ActionButtonProps> = ({
  icon,
  label,
  onClick,
  variant = 'water',
  darkMode = false,
}) => {
  const isOutlineStyle = variant === 'xp';
  const buttonStyle = isOutlineStyle
    ? xpOutlineStyle[darkMode ? 'dark' : 'light']
    : waterButtonStyle;

  const iconColor = isOutlineStyle ? 'text-nature' : '';

  return (
    <button
      onClick={onClick}
      className={`
        w-full flex flex-col sm:flex-row items-center justify-center gap-2 sm:gap-3
        px-4 sm:px-6 py-5 sm:py-4 rounded-3xl
        font-semibold text-sm sm:text-base
        transition-all duration-200
        ${buttonStyle}
        ${iconColor}
      `}
    >
      <span className="flex items-center justify-center text-2xl sm:text-xl">{icon}</span>
      <span>{label}</span>
    </button>
  );
};
