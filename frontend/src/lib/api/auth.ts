import { apiClient } from "./client";
import { LoginCredentials, RegisterData, Token } from "@/types";

/**
 * Backend user response
 */
interface UserResponse {
  id: number;
  username: string;
  email: string;
  xp: number;
  day_streak: number;
  location_city: string | null;
  created_at: string | null;
}

/**
 * Authentication API endpoints
 */
export const authApi = {
  /**
   * Login user
   * @param credentials - Username and password
   * @returns Access token
   */
  async login(credentials: LoginCredentials): Promise<Token> {
    const formData = new URLSearchParams();
    formData.append("username", credentials.username);
    formData.append("password", credentials.password);

    const response = await apiClient.post<Token>("/auth/login", formData, {
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
    });
    return response.data;
  },

  /**
   * Register new user
   * @param data - User registration data
   */
  async register(data: RegisterData): Promise<void> {
    await apiClient.post("/auth/register", {
      username: data.username,
      email: data.email,
      password: data.password,
    });
  },

  /**
   * Get current user info
   * Requires authentication
   */
  async getCurrentUser(): Promise<UserResponse> {
    const response = await apiClient.get<UserResponse>("/users/me");
    return response.data;
  },
};
