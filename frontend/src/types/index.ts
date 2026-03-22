/**
 * Core domain types for the Plant Care application
 */

/**
 * User profile and gamification data
 */
export interface UserData {
  name: string;
  level: number;
  xp: number;
  xpToNextLevel: number;
  streak: number;
  healthScore: number;
  weeklyChallenge: {
    completed: number;
    total: number;
    description: string;
  };
  achievements: Achievement[];
}

/**
 * Achievement/badge earned by user
 */
export interface Achievement {
  id: string;
  name: string;
  description: string;
  unlocked: boolean;
  icon: string;
}

/**
 * Authentication token response
 */
export interface Token {
  access_token: string;
  token_type: string;
}

/**
 * User authentication credentials
 */
export interface LoginCredentials {
  username: string;
  password: string;
}

/**
 * User registration data
 */
export interface RegisterData {
  username: string;
  email: string;
  password: string;
}

/**
 * Authenticated user information
 */
export interface User {
  id: string;
  username: string;
  email: string;
  name?: string;
  created_at: string;
}

/**
 * Plant identification result from AI
 */
export interface PlantIdentification {
  name: string;
  scientificName: string;
  confidence: number;
  careInstructions: string[];
  temperature: string;
  light: string;
  wateringFrequency: number;
}

/**
 * User plant as returned by the backend (GET /my-plants)
 */
export interface UserPlant {
  id: number;
  plant_catalog_id: number | null;
  custom_name: string | null;
  note: string | null;
  created_at: string;
  sprouted_at: string | null;
}

/**
 * Payload for creating a new user plant (POST /my-plants)
 */
export interface UserPlantCreate {
  plant_catalog_id?: number | null;
  custom_name?: string | null;
  note?: string | null;
  sprouted_at?: string | null;
}

/**
 * Payload for updating a user plant (PATCH /my-plants/:id)
 */
export interface UserPlantUpdate {
  plant_catalog_id?: number | null;
  custom_name?: string | null;
  note?: string | null;
  sprouted_at?: string | null;
}

/**
 * API error response
 */
export interface ApiError {
  message: string;
  detail?: string;
  status?: number;
}
