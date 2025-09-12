import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL, SUPABASE_ANON_KEY } from '../constants/config';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
    'apikey': SUPABASE_ANON_KEY,
    'Authorization': `Bearer ${SUPABASE_ANON_KEY}`, // Required for Edge Functions
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  async (config) => {
    const token = await AsyncStorage.getItem('authToken');
    if (token) {
      // User is logged in - use their token
      config.headers.Authorization = `Bearer ${token}`;
    }
    // If no user token, keep the anon key Authorization header (already set)
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle errors and token refresh
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      try {
        // Import authService dynamically to avoid circular dependency
        const authService = require('./auth').default;
        
        // Try to refresh token
        console.log('Token expired, attempting refresh...');
        const newToken = await authService.refreshToken();
        
        // Retry original request with new token
        originalRequest.headers.Authorization = `Bearer ${newToken}`;
        return api.request(originalRequest);
      } catch (refreshError) {
        console.error('Token refresh failed:', refreshError);
        // Clear tokens and user will need to re-authenticate
        await AsyncStorage.multiRemove(['authToken', 'refreshToken', 'user']);
        
        // Optionally, trigger a navigation to login screen
        // This would require passing navigation or using a global event emitter
        return Promise.reject(new Error('Session expired. Please log in again.'));
      }
      
      // Don't retry if we're already in auth flow
      if (originalRequest.url.includes('/auth-')) {
        return Promise.reject(error);
      }
    }
    
    return Promise.reject(error);
  }
);

export default api;