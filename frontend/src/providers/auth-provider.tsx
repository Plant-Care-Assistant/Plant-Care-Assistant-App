
"use client";

import { createContext, useContext, useState, useEffect, ReactNode, useCallback } from "react";
import { User, LoginCredentials, RegisterData } from "@/types";
import { AuthAPI, UsersAPI, setAccessToken, getStoredToken, setRefreshToken, getStoredRefreshToken } from "@/lib/apiClient";

interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (credentials: LoginCredentials) => Promise<void>;
  register: (data: RegisterData) => Promise<void>;
  logout: () => void;
  isLoading: boolean;
  isAuthenticated: boolean;
  error: string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshTimeout, setRefreshTimeout] = useState<NodeJS.Timeout | null>(null);
  const [refreshToken, setRefreshTokenState] = useState<string | null>(null);

  const fetchUser = useCallback(async () => {
    try {
      const userData = await UsersAPI.me();
      setUser({
        id: String(userData.id),
        username: userData.username,
        email: userData.email,
        created_at: new Date().toISOString(),
      });
    } catch {
      setUser(null);
    }
  }, []);

  const logout = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      await AuthAPI.logout();
    } catch {}
    setUser(null);
    setToken(null);
    setAccessToken(null);
    setRefreshToken(null);
    setRefreshTokenState(null);
    if (refreshTimeout) clearTimeout(refreshTimeout);
    setIsLoading(false);
  }, [refreshTimeout]);

  const refreshTokenFunc = useCallback(async () => {
    const storedRefresh = getStoredRefreshToken();
    if (!storedRefresh) {
      await logout();
      return;
    }
    try {
      const { RefreshAPI } = await import("@/lib/apiClient");
      const tokens = await RefreshAPI.refresh(storedRefresh);
      setAccessToken(tokens.access_token);
      setRefreshToken(tokens.refresh_token);
      setToken(tokens.access_token);
      setRefreshTokenState(tokens.refresh_token);
      await fetchUser();
    } catch {
      await logout();
    }
  }, [fetchUser, logout]);

  const scheduleRefresh = useCallback((expiresIn: number) => {
    if (refreshTimeout) clearTimeout(refreshTimeout);
    const timeout = setTimeout(refreshTokenFunc, Math.max(0, (expiresIn - 60) * 1000));
    setRefreshTimeout(timeout);
  }, [refreshTimeout, refreshTokenFunc]);

  const login = useCallback(async (credentials: LoginCredentials) => {
    setIsLoading(true);
    setError(null);
    try {
      const tokens = await AuthAPI.login(credentials);
      setAccessToken(tokens.access_token);
      setRefreshToken(tokens.refresh_token);
      setToken(tokens.access_token);
      setRefreshTokenState(tokens.refresh_token);
      await fetchUser();
    } catch (err: any) {
      let detail = err?.response?.data?.detail;
      let errorString = "Login failed";
      if (Array.isArray(detail)) {
        errorString = detail.map((e: any) => typeof e === 'object' ? JSON.stringify(e) : String(e)).join(' | ');
      } else if (typeof detail === 'object' && detail !== null) {
        errorString = JSON.stringify(detail);
      } else if (detail) {
        errorString = String(detail);
      } else if (err.message) {
        errorString = String(err.message);
      }
      setError(errorString);
      setUser(null);
      setToken(null);
      setAccessToken(null);
      setRefreshToken(null);
      setRefreshTokenState(null);
    } finally {
      setIsLoading(false);
    }
  }, [fetchUser]);

  const register = useCallback(async (data: RegisterData) => {
    setIsLoading(true);
    setError(null);
    try {
      const tokens = await AuthAPI.register(data);
      setAccessToken(tokens.access_token);
      setRefreshToken(tokens.refresh_token);
      setToken(tokens.access_token);
      setRefreshTokenState(tokens.refresh_token);
      await fetchUser();
    } catch (err: any) {
      let detail = err?.response?.data?.detail;
      let errorString = "Registration failed";
      if (Array.isArray(detail)) {
        errorString = detail.map((e: any) => typeof e === 'object' ? JSON.stringify(e) : String(e)).join(' | ');
      } else if (typeof detail === 'object' && detail !== null) {
        errorString = JSON.stringify(detail);
      } else if (detail) {
        errorString = String(detail);
      } else if (err.message) {
        errorString = String(err.message);
      }
      setError(errorString);
      setUser(null);
      setToken(null);
      setAccessToken(null);
      setRefreshToken(null);
      setRefreshTokenState(null);
    } finally {
      setIsLoading(false);
    }
  }, [fetchUser]);

  useEffect(() => {
    const stored = getStoredToken();
    const storedRefresh = getStoredRefreshToken();
    if (stored) {
      setAccessToken(stored);
      setToken(stored);
      if (storedRefresh) {
        setRefreshToken(storedRefresh);
        setRefreshTokenState(storedRefresh);
      }
      fetchUser().finally(() => setIsLoading(false));
    } else {
      setIsLoading(false);
    }
    return () => {
      if (refreshTimeout) clearTimeout(refreshTimeout);
    };
  }, [fetchUser, refreshTimeout]);

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
        error,
      }}
    >
      {children}

    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
