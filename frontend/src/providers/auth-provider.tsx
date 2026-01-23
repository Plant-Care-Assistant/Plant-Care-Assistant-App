"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { User, LoginCredentials, RegisterData } from "@/types";

interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (credentials: LoginCredentials) => Promise<void>;
  register: (data: RegisterData) => Promise<void>;
  logout: () => void;
  isLoading: boolean;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

/**
 * Authentication provider component
 * Manages user authentication state and token storage
 */
export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Mocked always-authenticated state for UI/UX work
  useEffect(() => {
    const mockUser: User = {
      id: "1",
      username: "dev-user",
      email: "dev@example.com",
      created_at: new Date().toISOString(),
    };

    setUser(mockUser);
    setToken("dev-token");
    setIsLoading(false);
  }, []);

  const login = async (_credentials: LoginCredentials) => {
    // No-op login: keep mocked session
    setUser((prev) =>
      prev ?? { id: "1", username: "dev-user", email: "dev@example.com", created_at: new Date().toISOString() }
    );
    setToken("dev-token");
  };

  const register = async (_data: RegisterData) => {
    // No-op register: immediately "logged in"
    setUser({ id: "1", username: "dev-user", email: "dev@example.com", created_at: new Date().toISOString() });
    setToken("dev-token");
  };

  const logout = () => {
    // Keep user logged in during UI work; no state change
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        login,
        register,
        logout,
        isLoading,
        isAuthenticated: !!token,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

/**
 * Hook to access authentication context
 * @throws Error if used outside AuthProvider
 */
export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
