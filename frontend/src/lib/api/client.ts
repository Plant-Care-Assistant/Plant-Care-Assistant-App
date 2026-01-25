import axios from "axios";

/**
 * Base API URL - configure via environment variable
 * Defaults to "/api" so Next.js/Nginx can proxy to the backend.
 *
 * If you set NEXT_PUBLIC_API_URL, include the "/api" suffix
 * (e.g. "http://localhost:8080/api").
 */
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL
  ? process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "").endsWith("/api")
    ? process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "")
    : `${process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "")}/api`
  : "/api";

/**
 * AI service base URL - configure via environment variable
 * Defaults to "/ai" so Next.js/Nginx can proxy to the AI service.
 */
const AI_BASE_URL = process.env.NEXT_PUBLIC_AI_URL
  ? process.env.NEXT_PUBLIC_AI_URL.replace(/\/$/, "")
  : "/ai";

/**
 * Axios instance for API requests
 * Automatically includes auth token from localStorage
 */
export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Add auth token to requests
apiClient.interceptors.request.use((config) => {
  const token =
    typeof window !== "undefined" ? localStorage.getItem("auth_token") : null;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle auth errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      if (typeof window !== "undefined") {
        localStorage.removeItem("auth_token");
        document.cookie = "auth_token=; Path=/; Max-Age=0; SameSite=Lax";
        window.location.href = "/auth/login";
      }
    }
    return Promise.reject(error);
  },
);

/**
 * Axios instance for AI requests (no auth required)
 */
export const aiClient = axios.create({
  baseURL: AI_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});
