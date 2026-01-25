import { apiClient } from "./client";
import { UserData } from "@/types";

interface UserPublic {
  id: number;
  username: string;
  email: string;
  xp: number;
  day_streak: number;
  location_city: string | null;
  created_at: string | null;
}

interface UserProfileUpdate {
  username?: string;
  email?: string;
  location_city?: string;
  preferences?: {
    dark_mode?: boolean;
    care_reminders?: boolean;
    weather_tips?: boolean;
  };
}

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
  async updateProfile(data: UserProfileUpdate): Promise<UserPublic> {
    const response = await apiClient.patch<UserPublic>("/users/me", data);
    return response.data;
  },
};
