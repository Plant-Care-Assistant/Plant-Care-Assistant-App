import { apiClient } from "./client";
import { LoginCredentials, RegisterData, Token } from "@/types";

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
    await apiClient.post("/users", {
      username: data.username,
      email: data.email,
      password: data.password,
    });
  },

  /**
   * Get current user info
   * Requires authentication
   */
  async getCurrentUser(): Promise<any> {
    const response = await apiClient.get("/users/me");
    return response.data;
  },
};
