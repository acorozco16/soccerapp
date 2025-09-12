import AsyncStorage from '@react-native-async-storage/async-storage';
import api from './api';
import { API_ENDPOINTS, SUPABASE_URL, SUPABASE_ANON_KEY } from '../constants/config';

// Real authentication with Supabase backend

class AuthService {
  async login(email, password) {
    try {
      const response = await api.post(API_ENDPOINTS.LOGIN, {
        email,
        password,
      });
      
      const { access_token, refresh_token, user } = response.data;
      
      // Store tokens and user info
      await AsyncStorage.setItem('authToken', access_token);
      if (refresh_token) {
        await AsyncStorage.setItem('refreshToken', refresh_token);
      }
      await AsyncStorage.setItem('user', JSON.stringify(user));
      
      return { success: true, user };
    } catch (error) {
      console.error('Login error:', error);
      console.error('Login error response:', error.response?.data);
      
      // Safely extract error message to prevent crashes
      let errorMessage = 'Login failed. Please try again.';
      try {
        if (error.response?.data?.detail && typeof error.response.data.detail === 'string') {
          errorMessage = error.response.data.detail;
        } else if (error.message && typeof error.message === 'string') {
          errorMessage = error.message;
        }
      } catch (e) {
        console.error('Error processing error message:', e);
      }
      
      return { 
        success: false, 
        error: errorMessage
      };
    }
  }

  async register(email, password, fullName) {
    try {
      const response = await api.post(API_ENDPOINTS.REGISTER, {
        email,
        password,
        full_name: fullName,
      });
      
      const { access_token, refresh_token, user } = response.data;
      
      // Store tokens and user info
      await AsyncStorage.setItem('authToken', access_token);
      if (refresh_token) {
        await AsyncStorage.setItem('refreshToken', refresh_token);
      }
      await AsyncStorage.setItem('user', JSON.stringify(user));
      
      return { success: true, user };
    } catch (error) {
      console.error('Registration error:', error);
      console.error('Registration error response:', error.response?.data);
      
      // Safely extract error message to prevent crashes
      let errorMessage = 'Registration failed. Please try again.';
      try {
        if (error.response?.data?.detail && typeof error.response.data.detail === 'string') {
          errorMessage = error.response.data.detail;
        } else if (error.message && typeof error.message === 'string') {
          errorMessage = error.message;
        }
      } catch (e) {
        console.error('Error processing error message:', e);
      }
      
      return { 
        success: false, 
        error: errorMessage
      };
    }
  }

  async logout() {
    try {
      await api.post(API_ENDPOINTS.LOGOUT);
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear local storage regardless
      await AsyncStorage.removeItem('authToken');
      await AsyncStorage.removeItem('user');
    }
  }

  async getCurrentUser() {
    try {
      const userStr = await AsyncStorage.getItem('user');
      return userStr ? JSON.parse(userStr) : null;
    } catch (error) {
      console.error('Get user error:', error);
      return null;
    }
  }

  // Check if token is expired or will expire soon
  async isTokenExpired(bufferMinutes = 5) {
    try {
      const token = await AsyncStorage.getItem('authToken');
      if (!token) return true;

      // Decode JWT to check expiration
      const base64Url = token.split('.')[1];
      const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
      const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
        return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
      }).join(''));

      const payload = JSON.parse(jsonPayload);
      const exp = payload.exp * 1000; // Convert to milliseconds
      const now = Date.now();
      const bufferTime = bufferMinutes * 60 * 1000; // Convert minutes to milliseconds

      return now + bufferTime >= exp;
    } catch (error) {
      console.error('Error checking token expiration:', error);
      return true; // Assume expired if we can't parse
    }
  }

  // Auto-refresh token if needed
  async ensureValidToken(forceRefresh = false) {
    try {
      console.log('ensureValidToken: Checking if token needs refresh...');
      const isExpired = await this.isTokenExpired();
      console.log('ensureValidToken: Token expired?', isExpired);
      
      // Force refresh for debugging - backend keeps rejecting tokens
      if (forceRefresh || isExpired) {
        console.log(forceRefresh ? 'Forcing token refresh for debugging...' : 'Token expired or expiring soon, refreshing...');
        const newToken = await this.refreshToken();
        console.log('ensureValidToken: Token refresh completed, new token:', newToken ? 'received' : 'failed');
        return !!newToken;
      }
      
      console.log('ensureValidToken: Token is still valid, no refresh needed');
      return true;
    } catch (error) {
      console.error('Failed to ensure valid token:', error);
      return false;
    }
  }

  async isAuthenticated() {
    const token = await AsyncStorage.getItem('authToken');
    return !!token;
  }

  async refreshToken() {
    try {
      console.log('refreshToken: Starting token refresh process...');
      
      // Try to refresh the token
      const refreshToken = await AsyncStorage.getItem('refreshToken');
      console.log('refreshToken: Retrieved refresh token:', refreshToken ? 'present' : 'missing');
      
      if (!refreshToken) {
        throw new Error('No refresh token available');
      }

      console.log('refreshToken: Making request to Supabase token endpoint...');
      
      // Use Supabase's direct auth API for token refresh
      const response = await fetch(`${SUPABASE_URL}/auth/v1/token?grant_type=refresh_token`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'apikey': SUPABASE_ANON_KEY,
        },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });

      console.log('refreshToken: Response status:', response.status);
      
      if (!response.ok) {
        const error = await response.json();
        console.error('refreshToken: Error response:', error);
        throw new Error(error.msg || 'Token refresh failed');
      }

      const data = await response.json();
      console.log('refreshToken: Success response received');
      console.log('refreshToken: Full response data keys:', Object.keys(data));
      // Security: Removed full response logging to prevent token exposure
      
      const { access_token, refresh_token: newRefreshToken } = data;
      // Security: Removed token substring logging to prevent token exposure
      console.log('refreshToken: Access token received:', !!access_token);

      if (!access_token) {
        throw new Error('No access token received from refresh');
      }

      console.log('refreshToken: Storing new tokens...');
      
      // Log token expiration for debugging
      try {
        const base64Url = access_token.split('.')[1];
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
          return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }).join(''));
        const payload = JSON.parse(jsonPayload);
        const exp = new Date(payload.exp * 1000);
        console.log('refreshToken: New token expires at:', exp.toISOString());
      } catch (e) {
        console.log('refreshToken: Could not decode new token expiration');
      }
      
      // Store new tokens
      await AsyncStorage.setItem('authToken', access_token);
      if (newRefreshToken) {
        await AsyncStorage.setItem('refreshToken', newRefreshToken);
        console.log('refreshToken: Updated both access and refresh tokens');
      } else {
        console.log('refreshToken: Updated access token only');
      }

      console.log('Auth token refreshed successfully');
      return access_token;
    } catch (error) {
      console.error('Token refresh failed:', error);
      // Clear invalid tokens
      await AsyncStorage.removeItem('authToken');
      await AsyncStorage.removeItem('refreshToken');
      throw error;
    }
  }
}

export default new AuthService();