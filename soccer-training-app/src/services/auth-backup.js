// Temporary mock auth service for testing
import AsyncStorage from '@react-native-async-storage/async-storage';

class MockAuthService {
  async login(email, password) {
    try {
      // Mock successful login for testing
      if (email && password) {
        const mockUser = {
          id: '12345',
          email: email,
          full_name: 'Test User',
          created_at: new Date().toISOString()
        };
        
        const mockToken = 'mock-jwt-token-for-testing';
        
        // Store token and user info
        await AsyncStorage.setItem('authToken', mockToken);
        await AsyncStorage.setItem('user', JSON.stringify(mockUser));
        
        return { success: true, user: mockUser };
      } else {
        return { success: false, error: 'Please enter email and password' };
      }
    } catch (error) {
      return { success: false, error: 'Login failed. Please try again.' };
    }
  }

  async register(email, password, fullName) {
    try {
      // Mock successful registration for testing
      if (email && password && fullName) {
        const mockUser = {
          id: '12345',
          email: email,
          full_name: fullName,
          created_at: new Date().toISOString()
        };
        
        const mockToken = 'mock-jwt-token-for-testing';
        
        // Store token and user info
        await AsyncStorage.setItem('authToken', mockToken);
        await AsyncStorage.setItem('user', JSON.stringify(mockUser));
        
        return { success: true, user: mockUser };
      } else {
        return { success: false, error: 'Please fill in all fields' };
      }
    } catch (error) {
      return { success: false, error: 'Registration failed. Please try again.' };
    }
  }

  async logout() {
    await AsyncStorage.removeItem('authToken');
    await AsyncStorage.removeItem('user');
  }

  async getCurrentUser() {
    try {
      const userStr = await AsyncStorage.getItem('user');
      return userStr ? JSON.parse(userStr) : null;
    } catch (error) {
      return null;
    }
  }

  async isAuthenticated() {
    const token = await AsyncStorage.getItem('authToken');
    return !!token;
  }
}

export default new MockAuthService();