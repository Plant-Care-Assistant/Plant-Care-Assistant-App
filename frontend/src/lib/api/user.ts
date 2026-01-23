import { apiClient } from "./client";
import { UserData } from "@/types";

/**
 * User profile and gamification API endpoints
 */
export const userApi = {
  /**
   * Get user gamification data
   */
  async getUserData(): Promise<UserData> {
    const response = await apiClient.get<UserData>("/users/me/stats");
    return response.data;
  },

  /**
   * Update user profile
   */
  async updateProfile(data: Partial<UserData>): Promise<UserData> {
    const response = await apiClient.patch<UserData>("/users/me", data);
    return response.data;
  },
};
