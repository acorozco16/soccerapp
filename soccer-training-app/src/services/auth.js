import AsyncStorage from '@react-native-async-storage/async-storage';
import api from './api';
import { API_ENDPOINTS } from '../constants/config';

// Real authentication with Supabase backend

class AuthService {
  async login(email, password) {
    try {
      const response = await api.post(API_ENDPOINTS.LOGIN, {
        email,
        password,
      });
      
      const { access_token, user } = response.data;
      
      // Store token and user info
      await AsyncStorage.setItem('authToken', access_token);
      await AsyncStorage.setItem('user', JSON.stringify(user));
      
      return { success: true, user };
    } catch (error) {
      console.error('Login error:', error);
      console.error('Login error response:', error.response?.data);
      return { 
        success: false, 
        error: error.response?.data?.detail || 'Login failed. Please try again.' 
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
      
      const { access_token, user } = response.data;
      
      // Store token and user info
      await AsyncStorage.setItem('authToken', access_token);
      await AsyncStorage.setItem('user', JSON.stringify(user));
      
      return { success: true, user };
    } catch (error) {
      console.error('Registration error:', error);
      console.error('Registration error response:', error.response?.data);
      return { 
        success: false, 
        error: error.response?.data?.detail || 'Registration failed. Please try again.' 
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

  async isAuthenticated() {
    const token = await AsyncStorage.getItem('authToken');
    return !!token;
  }
}

export default new AuthService();