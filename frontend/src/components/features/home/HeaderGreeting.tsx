interface HeaderGreetingProps {
  darkMode: boolean;
}

export function HeaderGreeting({ darkMode }: HeaderGreetingProps) {
  return (
    <div>
      <h1 className={`text-3xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
        Hi, Sarah! 👋
      </h1>
      <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
        Let's take care of your plants today
      </p>
    </div>
  );
}
