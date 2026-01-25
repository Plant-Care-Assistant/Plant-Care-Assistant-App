// Plant Care Assistant API Client
// src/lib/apiClient.ts

import axios, { AxiosError, AxiosInstance, AxiosRequestConfig, AxiosResponse, InternalAxiosRequestConfig } from 'axios';

// =========================
// TypeScript Interfaces (match backend Pydantic models)
// =========================

export interface UserPublic {
  id: number;
  email: string;
  username: string;
  preferences?: UserPreferences;
  // Add more fields as needed
}

export interface UserPreferences {
  // Define user preferences fields
  [key: string]: any;
}

export interface UserUpdate {
  username?: string;
  preferences?: UserPreferences;
}

export interface PlantPublic {
  id: number;
  name: string;
  description?: string;
  // Add more fields as needed
}

export interface PageParams {
  limit?: number;
  offset?: number;
}

export interface UserPlantPublic {
  id: number;
  plant_id: number;
  nickname?: string;
  sprouted_at?: string;
  // Add more fields as needed
}

export interface UserPlantCreate {
  plant_id: number;
  nickname?: string;
  sprouted_at?: string;
}

export interface UserPlantUpdate {
  nickname?: string;
  sprouted_at?: string;
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface AuthCredentials {
  username: string;
  password: string;
}

// =========================
// Axios Instance & Interceptors
// =========================

// Always use relative URL so requests go through nginx proxy
const apiBaseURL = '/api';


let accessToken: string | null = null;
let refreshToken: string | null = null;


// Store token in memory and localStorage
export function setAccessToken(token: string | null) {
  accessToken = token;
  if (typeof window !== 'undefined') {
    if (token) {
      localStorage.setItem('access_token', token);
    } else {
      localStorage.removeItem('access_token');
    }
  }
}

export function setRefreshToken(token: string | null) {
  refreshToken = token;
  if (typeof window !== 'undefined') {
    if (token) {
      localStorage.setItem('refresh_token', token);
    } else {
      localStorage.removeItem('refresh_token');
    }
  }
}

// Retrieve token from localStorage

export function getStoredToken(): string | null {
  if (typeof window !== 'undefined') {
    return localStorage.getItem('access_token');
  }
  return null;
}

export function getStoredRefreshToken(): string | null {
  if (typeof window !== 'undefined') {
    return localStorage.getItem('refresh_token');
  }
  return null;
}

const api: AxiosInstance = axios.create({
  baseURL: apiBaseURL,
  withCredentials: true,
});

// Request interceptor: Attach JWT
api.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  if (accessToken && config.headers) {
    config.headers['Authorization'] = `Bearer ${accessToken}`;
  }
  return config;
});


// Token refresh concurrency control
let isRefreshing = false;
let refreshSubscribers: ((token: string) => void)[] = [];

function subscribeTokenRefresh(cb: (token: string) => void) {
  refreshSubscribers.push(cb);
}
function onRefreshed(token: string) {
  refreshSubscribers.forEach((cb) => cb(token));
  refreshSubscribers = [];
}

api.interceptors.response.use(
  (response: AxiosResponse) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean };
    if (error.response && error.response.status === 401 && !originalRequest._retry) {
      if (!refreshToken) {
        refreshToken = getStoredRefreshToken();
      }
      if (!refreshToken) {
        // No refresh token, logout
        return Promise.reject(error);
      }
      if (isRefreshing) {
        // Wait for refresh to complete
        return new Promise((resolve, reject) => {
          subscribeTokenRefresh((newToken) => {
            if (!newToken) return reject(error);
            originalRequest.headers = originalRequest.headers || {};
            originalRequest.headers['Authorization'] = `Bearer ${newToken}`;
            originalRequest._retry = true;
            resolve(api(originalRequest));
          });
        });
      }
      isRefreshing = true;
      originalRequest._retry = true;
      try {
        const tokens = await RefreshAPI.refresh(refreshToken);
        setAccessToken(tokens.access_token);
        setRefreshToken(tokens.refresh_token);
        accessToken = tokens.access_token;
        refreshToken = tokens.refresh_token;
        onRefreshed(tokens.access_token);
        originalRequest.headers = originalRequest.headers || {};
        originalRequest.headers['Authorization'] = `Bearer ${tokens.access_token}`;
        return api(originalRequest);
      } catch (refreshError) {
        setAccessToken(null);
        setRefreshToken(null);
        onRefreshed("");
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }
    return Promise.reject(error);
  }
);
// =========================
// Refresh API
// =========================

export const RefreshAPI = {
  async refresh(refreshToken: string): Promise<AuthTokens> {
    const res = await axios.post<AuthTokens>(`${apiBaseURL}/auth/refresh`, { refresh_token: refreshToken });
    return res.data;
  },
};

// =========================
// API Methods
// =========================

export const AuthAPI = {
  async login(data: AuthCredentials): Promise<AuthTokens> {
    // Send as form data for FastAPI OAuth2 compatibility
    const params = new URLSearchParams();
    params.append('username', data.username);
    params.append('password', data.password);
    const res = await api.post<AuthTokens>('/auth/login', params, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });
    return res.data;
  },
  async logout(): Promise<void> {
    await api.post('/auth/logout');
  },
  async register(data: { email: string; username: string; password: string }): Promise<AuthTokens> {
    const res = await api.post<AuthTokens>('/auth/register', data);
    return res.data;
  },
  // Add refresh, etc. as needed
};

export const PlantsAPI = {
    async addToCollection(plantId: number): Promise<void> {
      await api.post(`/my-plants`, { plant_id: plantId });
    },
  async list(params?: PageParams): Promise<PlantPublic[]> {
    const res = await api.get<PlantPublic[]>('/plants', { params });
    return res.data;
  },
  async search(q: string): Promise<PlantPublic[]> {
    const res = await api.get<PlantPublic[]>('/plants/search', { params: { q } });
    return res.data;
  },
  async get(plantId: number): Promise<PlantPublic> {
    const res = await api.get<PlantPublic>(`/plants/${plantId}`);
    return res.data;
  },
  async getImage(plantId: number): Promise<Blob> {
    const res = await api.get(`/plants/${plantId}/image`, { responseType: 'blob' });
    return res.data;
  },
};

export const UserPlantsAPI = {
  async list(): Promise<UserPlantPublic[]> {
    const res = await api.get<UserPlantPublic[]>('/my-plants');
    return res.data;
  },
  async create(data: UserPlantCreate): Promise<UserPlantPublic> {
    const res = await api.post<UserPlantPublic>('/my-plants', data);
    return res.data;
  },
  async get(plantId: number): Promise<UserPlantPublic> {
    const res = await api.get<UserPlantPublic>(`/my-plants/${plantId}`);
    return res.data;
  },
  async update(plantId: number, data: UserPlantUpdate): Promise<UserPlantPublic> {
    const res = await api.patch<UserPlantPublic>(`/my-plants/${plantId}`, data);
    return res.data;
  },
  async delete(plantId: number): Promise<void> {
    await api.delete(`/my-plants/${plantId}`);
  },
  async uploadImage(plantId: number, file: File): Promise<void> {
    const formData = new FormData();
    formData.append('file', file);
    await api.post(`/my-plants/${plantId}/image`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  async getImage(plantId: number): Promise<Blob> {
    const res = await api.get(`/my-plants/${plantId}/image`, { responseType: 'blob' });
    return res.data;
  },
  async deleteImage(plantId: number): Promise<void> {
    await api.delete(`/my-plants/${plantId}/image`);
  },
};

export const UsersAPI = {
  async list(): Promise<UserPublic[]> {
    const res = await api.get<UserPublic[]>('/users');
    return res.data;
  },
  async me(): Promise<UserPublic> {
    const res = await api.get<UserPublic>('/users/me');
    return res.data;
  },
  async updateMe(data: UserUpdate): Promise<UserPublic> {
    const res = await api.patch<UserPublic>('/users/me', data);
    return res.data;
  },
  async deleteMe(): Promise<void> {
    await api.delete('/users/me');
  },
  async getSettings(): Promise<UserPreferences> {
    const res = await api.get<UserPreferences>('/users/me/settings');
    return res.data;
  },
  async updateSettings(data: UserPreferences): Promise<UserPreferences> {
    const res = await api.put<UserPreferences>('/users/me/settings', data);
    return res.data;
  },
};

// =========================
// AI Inference API
// =========================

export interface PlantPrediction {
  predictions: Array<{
    class_id: string;
    class_name: string;
    confidence: number;
  }>;
  processing_time_ms: number;
}

export const AIAPI = {
  async identifyPlant(imageFile: File): Promise<PlantPrediction> {
    const formData = new FormData();
    formData.append('file', imageFile);

    // Always use relative URL so requests go through nginx proxy
    const res = await axios.post<PlantPrediction>(`/api/ai/predict`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },
};

export default api;
