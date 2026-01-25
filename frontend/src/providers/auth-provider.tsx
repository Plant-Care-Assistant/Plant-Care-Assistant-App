"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
  useCallback,
} from "react";
import { AxiosError } from "axios";
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
  error: string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const TOKEN_KEY = "auth_token";
const COOKIE_KEY = "auth_token";

function setAuthCookie(token: string | null) {
  if (typeof document === "undefined") {
    return;
  }
  if (token) {
    document.cookie = `${COOKIE_KEY}=${token}; Path=/; SameSite=Lax`;
  } else {
    document.cookie = `${COOKIE_KEY}=; Path=/; Max-Age=0; SameSite=Lax`;
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const initAuth = async () => {
      const storedToken = localStorage.getItem(TOKEN_KEY);
      if (storedToken) {
        setToken(storedToken);
        setAuthCookie(storedToken);
        try {
          const currentUser = await authApi.getCurrentUser();
          setUser({
            id: String(currentUser.id),
            username: currentUser.username,
            email: currentUser.email,
            created_at: currentUser.created_at ?? new Date().toISOString(),
          });
        } catch {
          localStorage.removeItem(TOKEN_KEY);
          setToken(null);
        }
      }
      setIsLoading(false);
    };

    initAuth();
  }, []);

  const login = useCallback(async (credentials: LoginCredentials) => {
    setError(null);
    setIsLoading(true);
    try {
      const tokenResponse = await authApi.login(credentials);
      const accessToken = tokenResponse.access_token;
      localStorage.setItem(TOKEN_KEY, accessToken);
      setToken(accessToken);
      setAuthCookie(accessToken);

      const currentUser = await authApi.getCurrentUser();
      setUser({
        id: String(currentUser.id),
        username: currentUser.username,
        email: currentUser.email,
        created_at: currentUser.created_at ?? new Date().toISOString(),
      });
    } catch (err) {
      const axiosError = err as AxiosError<{ detail?: string }>;
      const message =
        axiosError.response?.data?.detail || "Login failed. Please try again.";
      setError(message);
      throw new Error(message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const register = useCallback(
    async (data: RegisterData) => {
      setError(null);
      setIsLoading(true);
      try {
        await authApi.register(data);
        await login({ username: data.email, password: data.password });
      } catch (err) {
        const axiosError = err as AxiosError<{ detail?: string }>;
        const message =
          axiosError.response?.data?.detail ||
          "Registration failed. Please try again.";
        setError(message);
        throw new Error(message);
      } finally {
        setIsLoading(false);
      }
    },
    [login],
  );

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    setToken(null);
    setUser(null);
    setError(null);
    setAuthCookie(null);
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
        isAuthenticated: !!token && !!user,
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
