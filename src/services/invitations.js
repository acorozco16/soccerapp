import api from './api';
import { API_BASE_URL } from '../constants/config';
import AsyncStorage from '@react-native-async-storage/async-storage';
import axios from 'axios';

// Create API client for invitations
const inviteApi = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
inviteApi.interceptors.request.use(
  async (config) => {
    const token = await AsyncStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

class InvitationService {
  
  /**
   * Send an invitation to a friend via email
   */
  async sendInvitation(email, phone = null) {
    try {
      const response = await inviteApi.post('/invitations/send', {
        email,
        phone
      });
      
      return {
        success: true,
        data: response.data,
        message: response.data.message || 'Invitation sent successfully'
      };
      
    } catch (error) {
      console.error('Failed to send invitation:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to send invitation'
      };
    }
  }

  /**
   * Validate an invitation code
   */
  async validateInvitation(code) {
    try {
      const response = await inviteApi.get(`/invitations/validate/${code}`);
      
      return {
        success: true,
        data: response.data
      };
      
    } catch (error) {
      console.error('Failed to validate invitation:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Invalid invitation code'
      };
    }
  }

  /**
   * Accept an invitation during registration
   */
  async acceptInvitation(code, userId) {
    try {
      const response = await inviteApi.post(`/invitations/accept/${code}`, {
        user_id: userId
      });
      
      return {
        success: true,
        data: response.data
      };
      
    } catch (error) {
      console.error('Failed to accept invitation:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to accept invitation'
      };
    }
  }

  /**
   * Get connection requests (pending friend requests)
   */
  async getConnectionRequests() {
    try {
      const response = await inviteApi.get('/connections/requests');
      
      return {
        success: true,
        data: response.data
      };
      
    } catch (error) {
      console.error('Failed to get connection requests:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to get connection requests'
      };
    }
  }

  /**
   * Send a connection request to another user
   */
  async sendConnectionRequest(userId) {
    try {
      const response = await inviteApi.post('/connections/request', null, {
        params: { addressee_id: userId }
      });
      
      return {
        success: true,
        data: response.data
      };
      
    } catch (error) {
      console.error('Failed to send connection request:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to send connection request'
      };
    }
  }

  /**
   * Accept a connection request
   */
  async acceptConnectionRequest(connectionId) {
    try {
      const response = await inviteApi.put(`/connections/${connectionId}/accept`);
      
      return {
        success: true,
        data: response.data
      };
      
    } catch (error) {
      console.error('Failed to accept connection request:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to accept connection request'
      };
    }
  }

  /**
   * Get list of friends/connections
   */
  async getFriends() {
    try {
      const response = await inviteApi.get('/connections/friends');
      
      return {
        success: true,
        data: response.data
      };
      
    } catch (error) {
      console.error('Failed to get friends:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to get friends list'
      };
    }
  }

  /**
   * Remove a connection (unfriend)
   */
  async removeConnection(connectionId) {
    try {
      const response = await inviteApi.delete(`/connections/${connectionId}`);
      
      return {
        success: true,
        data: response.data
      };
      
    } catch (error) {
      console.error('Failed to remove connection:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to remove connection'
      };
    }
  }

  /**
   * Search for users by email or name
   */
  async searchUsers(query) {
    try {
      const response = await inviteApi.get('/connections/search', {
        params: { query }
      });
      
      return {
        success: true,
        data: response.data
      };
      
    } catch (error) {
      console.error('Failed to search users:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to search users'
      };
    }
  }
}

export default new InvitationService();