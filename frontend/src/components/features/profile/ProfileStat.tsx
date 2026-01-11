'use client';

interface ProfileStatProps {
  value: string | number;
  label: string;
  darkMode?: boolean;
}

export const ProfileStat: React.FC<ProfileStatProps> = ({ value, label, darkMode = false }) => {
  return (
    <div className={`flex-1 rounded-2xl shadow-sm p-4 sm:p-6 text-center ${
      darkMode ? 'bg-neutral-800' : 'bg-white'
    }`}>
      <div className={`text-2xl sm:text-3xl font-bold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
        {value}
      </div>
      <div className={`text-xs sm:text-sm mt-2 ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
        {label}
      </div>
    </div>
  );
};
