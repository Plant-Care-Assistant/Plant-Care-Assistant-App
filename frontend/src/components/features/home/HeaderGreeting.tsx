"use client";

import { useAuth } from "@/providers";

interface HeaderGreetingProps {
  darkMode: boolean;
}

export function HeaderGreeting({ darkMode }: HeaderGreetingProps) {
  const { user } = useAuth();
  const displayName = user?.username || "there";

  return (
    <div>
      <h1
        className={`text-3xl font-bold ${
          darkMode ? "text-white" : "text-gray-900"
        }`}
      >
        Hi, {displayName}! 👋
      </h1>
      <p className={`text-sm ${darkMode ? "text-gray-400" : "text-gray-600"}`}>
        Let's take care of your plants today
      </p>
    </div>
  );
}
