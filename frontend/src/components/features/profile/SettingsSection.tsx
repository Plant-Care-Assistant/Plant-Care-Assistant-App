import { Moon, Bell, Cloud } from 'lucide-react';
import { SettingItem } from './SettingItem';

interface SettingsSectionProps {
  darkMode?: boolean;
  onDarkModeToggle?: (enabled: boolean) => void;
  careRemindersEnabled: boolean;
  onCareRemindersToggle: (enabled: boolean) => void;
  weatherTipsEnabled: boolean;
  onWeatherTipsToggle: (enabled: boolean) => void;
}

export function SettingsSection({
  darkMode = false,
  onDarkModeToggle,
  careRemindersEnabled,
  onCareRemindersToggle,
  weatherTipsEnabled,
  onWeatherTipsToggle,
}: SettingsSectionProps) {
  return (
    <div className={`rounded-3xl shadow-lg p-6 sm:p-8 ${darkMode ? 'bg-neutral-800' : 'bg-white'}`}>
      <h2 className={`text-xl sm:text-2xl font-bold mb-6 sm:mb-8 ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
        Settings
      </h2>

      <div className="space-y-5 sm:space-y-6">
        <SettingItem
          icon={<Moon size={24} className="text-white" />}
          title="Dark Mode"
          description="Easier on the eyes"
          bgColor="bg-primary"
          enabled={darkMode}
          onToggle={onDarkModeToggle || (() => {})}
          darkMode={darkMode}
        />
        <SettingItem
          icon={<Bell size={24} className="text-white" />}
          title="Care Reminders"
          description="Daily notifications"
          bgColor="bg-accent2"
          enabled={careRemindersEnabled}
          onToggle={onCareRemindersToggle}
          darkMode={darkMode}
        />
        <SettingItem
          icon={<Cloud size={24} className="text-white" />}
          title="Weather Tips"
          description="Smart care suggestions"
          bgColor="bg-accent"
          enabled={weatherTipsEnabled}
          onToggle={onWeatherTipsToggle}
          darkMode={darkMode}
        />
      </div>
    </div>
  );
}
