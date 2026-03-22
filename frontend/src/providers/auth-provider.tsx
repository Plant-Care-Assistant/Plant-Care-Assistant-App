"use client";

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from "react";
import { User, LoginCredentials, RegisterData } from "@/types";
import { authApi } from "@/lib/api/auth";

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

function setTokenCookie(token: string) {
  document.cookie = `auth_token=${token}; path=/; max-age=${60 * 60 * 24 * 28}; SameSite=Lax`;
}

function removeTokenCookie() {
  document.cookie = "auth_token=; path=/; max-age=0";
}

/**
 * Authentication provider component
 * Manages user authentication state and token storage
 */
export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // On mount, check for existing token and fetch user
  useEffect(() => {
    const savedToken = localStorage.getItem("auth_token");
    if (savedToken) {
      setToken(savedToken);
      setTokenCookie(savedToken);
      authApi
        .getCurrentUser()
        .then((userData) => setUser(userData))
        .catch(() => {
          // Token is invalid or expired — clear it
          localStorage.removeItem("auth_token");
          removeTokenCookie();
          setToken(null);
        })
        .finally(() => setIsLoading(false));
    } else {
      setIsLoading(false);
    }
  }, []);

  const login = useCallback(async (credentials: LoginCredentials) => {
    const tokenData = await authApi.login(credentials);
    const accessToken = tokenData.access_token;

    localStorage.setItem("auth_token", accessToken);
    setTokenCookie(accessToken);
    setToken(accessToken);

    const userData = await authApi.getCurrentUser();
    setUser(userData);
  }, []);

  const register = useCallback(async (data: RegisterData) => {
    await authApi.register(data);
    // Auto-login after registration
    await login({ username: data.email, password: data.password });
  }, [login]);

  const logout = useCallback(() => {
    localStorage.removeItem("auth_token");
    removeTokenCookie();
    setToken(null);
    setUser(null);
    window.location.href = "/auth/login";
  }, []);

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
