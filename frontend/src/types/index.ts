/**
 * Core domain types for the Plant Care application
 */

/**
 * Represents a plant in the user's collection
 */
export interface Plant {
  id: number;
  name: string;
  scientificName: string;
  image: string;
  health: number;
  streak: number;
  lastWatered: string;
  wateringFrequency: number;
  temperature: string;
  light: string;
  careInstructions: string[];
}

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
  refresh_token?: string;
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
 * API error response
 */
export interface ApiError {
  message: string;
  detail?: string;
  status?: number;
}
