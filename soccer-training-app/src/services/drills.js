import api from './api';
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import authService from './auth';
import { API_ENDPOINTS, DRILL_BASE_URL, API_BASE_URL } from '../constants/config';

// Create separate API client for drill server
const drillApi = axios.create({
  baseURL: DRILL_BASE_URL,
  timeout: 120000, // 2 minutes for video uploads
  headers: {
    'Content-Type': 'application/json',
  },
});

// Auth token management with auto-refresh
drillApi.interceptors.request.use(
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

// Auto-refresh auth tokens on 401 errors
drillApi.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      console.log('Token expired, attempting refresh...');
      
      try {
        // Attempt to refresh token
        await refreshAuthToken();
        
        // Get new token and retry original request
        const newToken = await AsyncStorage.getItem('authToken');
        if (newToken) {
          originalRequest.headers.Authorization = `Bearer ${newToken}`;
          console.log('Token refreshed, retrying original request');
          return drillApi.request(originalRequest);
        }
      } catch (refreshError) {
        console.error('Token refresh failed:', refreshError);
        // Clear invalid token and redirect to login
        await AsyncStorage.removeItem('authToken');
        // Note: Navigation should be handled by the calling component
        return Promise.reject(new Error('Session expired. Please log in again.'));
      }
    }
    
    return Promise.reject(error);
  }
);

// Refresh authentication token
const refreshAuthToken = async () => {
  try {
    // Use the authService refresh method which handles everything properly
    const newToken = await authService.refreshToken();
    console.log('Auth token refreshed successfully via authService');
    return newToken;
  } catch (error) {
    console.error('Auth token refresh failed:', error);
    throw error;
  }
};

class DrillService {
  async getAvailableDrills() {
    try {
      const response = await drillApi.get(API_ENDPOINTS.AVAILABLE_DRILLS);
      return {
        success: true,
        drills: response.data.drills || [],
        total: response.data.total_count || 0
      };
    } catch (error) {
      console.error('Failed to fetch drills:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to load drills'
      };
    }
  }

  async getDrillInfo(drillType) {
    try {
      const response = await drillApi.get(`/drill/info/${drillType}`);
      return {
        success: true,
        drill: response.data
      };
    } catch (error) {
      console.error('Failed to fetch drill info:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to load drill information'
      };
    }
  }

  async getDrillBenchmark(drillType) {
    try {
      const response = await drillApi.get(`/drill/benchmark/${drillType}`);
      return {
        success: true,
        benchmark: response.data
      };
    } catch (error) {
      console.error('Failed to fetch drill benchmark:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to load benchmark data'
      };
    }
  }

  async startDrillAnalysis(drillType, videoFile) {
    try {
      // Validate video file before upload
      if (!videoFile || !videoFile.uri) {
        throw new Error('Invalid video file');
      }

      const formData = new FormData();
      formData.append('file', {
        uri: videoFile.uri,
        type: 'video/mp4',
        name: 'drill_video.mp4'
      });
      formData.append('drill_type', drillType);

      const response = await drillApi.post(API_ENDPOINTS.ANALYZE_DRILL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutes for video upload
      });

      return {
        success: true,
        analysisId: response.data.analysis_id,
        drillType: response.data.drill_type
      };
    } catch (error) {
      console.error('Failed to start drill analysis:', error);
      
      // Better error messages based on error type
      let errorMessage = 'Failed to start analysis';
      
      if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
        errorMessage = 'Upload timed out. Please check your connection and try again.';
      } else if (error.response?.status === 413) {
        errorMessage = 'Video file too large. Try recording a shorter video.';
      } else if (error.response?.status >= 500) {
        errorMessage = 'Server error. Please try again in a few moments.';
      } else if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      }
      
      return {
        success: false,
        error: errorMessage
      };
    }
  }

  // New method with retry logic and progress tracking
  async startDrillAnalysisWithRetry(drillType, videoFile, onProgress, maxRetries = 3) {
    console.log('Starting upload with retry mechanism, max retries:', maxRetries);
    
    return await this.uploadWithRetry(
      () => this.uploadWithProgress(drillType, videoFile, onProgress),
      maxRetries
    );
  }

  // Upload with real progress tracking using XMLHttpRequest
  async uploadWithProgress(drillType, videoFile, onProgress) {
    return new Promise(async (resolve, reject) => {
      try {
        // Ensure token is valid before upload
        const tokenValid = await authService.ensureValidToken();
        if (!tokenValid) {
          reject(new Error('Unable to refresh authentication token. Please log in again.'));
          return;
        }

        // Get current auth token
        const supabaseToken = await AsyncStorage.getItem('authToken');
        console.log('Retrieved Supabase token:', supabaseToken ? 'present' : 'missing');
        if (!supabaseToken) {
          reject(new Error('No authentication token available. Please log in again.'));
          return;
        }

        // Try to get or exchange for drill server token
        let drillToken = await AsyncStorage.getItem('drillToken');
        
        if (!drillToken) {
          console.log('No drill token found, attempting exchange...');
          try {
            // Attempt token exchange (if endpoint exists)
            const response = await fetch(`${DRILL_BASE_URL}/auth/exchange`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${supabaseToken}`
              },
              body: JSON.stringify({ supabase_token: supabaseToken })
            });
            
            if (response.ok) {
              const data = await response.json();
              drillToken = data.access_token; // Fixed: use access_token instead of drill_token
              await AsyncStorage.setItem('drillToken', drillToken);
              console.log('Successfully exchanged for drill token');
            } else {
              console.log('Token exchange failed (status:', response.status, '). Using Supabase token directly.');
              drillToken = supabaseToken;
            }
          } catch (error) {
            console.log('Token exchange endpoint not available:', error.message);
            console.log('Using Supabase token directly - drill server needs to be configured to accept Supabase JWTs');
            drillToken = supabaseToken;
          }
        }

        const token = drillToken;

        const formData = new FormData();
        
        // Fix file object format for React Native
        const fileObject = {
          uri: videoFile.uri,
          type: 'video/mp4',
          name: 'drill_video.mp4'
        };
        
        console.log('Video file object:', fileObject);
        console.log('Drill type:', drillType);
        
        formData.append('file', fileObject);
        // drill_type is now sent as query parameter in URL
        
        console.log('FormData created successfully');

        const xhr = new XMLHttpRequest();

        // Track request lifecycle
        xhr.addEventListener('loadstart', () => {
          console.log('XMLHttpRequest loadstart - request initiated');
        });

        xhr.addEventListener('loadend', () => {
          console.log('XMLHttpRequest loadend - request completed');
          console.log('Final status:', xhr.status);
          console.log('Final readyState:', xhr.readyState);
        });

        // Track upload progress
        xhr.upload.addEventListener('progress', (event) => {
          if (event.lengthComputable && onProgress) {
            const progress = (event.loaded / event.total) * 100;
            console.log('Upload progress:', Math.round(progress) + '%');
            onProgress(progress);
          }
        });

        // Handle successful upload
        xhr.addEventListener('load', () => {
          console.log('XMLHttpRequest load event - status:', xhr.status);
          console.log('XMLHttpRequest load event - statusText:', xhr.statusText);
          console.log('XMLHttpRequest load event - responseText:', xhr.responseText);
          
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              const response = JSON.parse(xhr.responseText);
              resolve({
                success: true,
                analysisId: response.analysis_id,
                drillType: response.drill_type
              });
            } catch (parseError) {
              console.error('Parse error in successful response:', parseError);
              reject(new Error('Invalid server response'));
            }
          } else {
            let errorMessage = `Upload failed with status ${xhr.status}`;
            
            try {
              const errorData = JSON.parse(xhr.responseText);
              errorMessage = errorData.detail || errorMessage;
            } catch (parseError) {
              // Use status-based error messages
              if (xhr.status === 413) {
                errorMessage = 'Video file too large. Try recording a shorter video.';
              } else if (xhr.status >= 500) {
                errorMessage = 'Server error. Please try again in a few moments.';
              } else if (xhr.status === 401) {
                const responseText = xhr.responseText;
                if (responseText.includes('JWT') || responseText.includes('token')) {
                  errorMessage = 'Authentication token format incompatible with drill server. Please contact support.';
                } else {
                  errorMessage = 'Authentication failed. Please log in again.';
                }
              } else if (xhr.status === 422) {
                errorMessage = `Invalid request format (422). Server response: ${xhr.responseText}`;
              } else {
                errorMessage = `Server returned status ${xhr.status}: ${xhr.statusText}`;
              }
            }
            
            console.error('Upload failed in load event:', errorMessage);
            reject(new Error(errorMessage));
          }
        });

        // Handle network errors
        xhr.addEventListener('error', (event) => {
          console.error('XMLHttpRequest error event:', event);
          console.error('XHR status:', xhr.status);
          console.error('XHR statusText:', xhr.statusText);
          console.error('XHR readyState:', xhr.readyState);
          console.error('XHR responseText:', xhr.responseText);
          reject(new Error(`Network error (status: ${xhr.status}). Please check your connection and try again.`));
        });

        // Handle timeout
        xhr.addEventListener('timeout', (event) => {
          console.error('XMLHttpRequest timeout event:', event);
          console.error('XHR status at timeout:', xhr.status);
          reject(new Error('Upload timed out. Please check your connection and try again.'));
        });

        // Configure and send request with drill_type as query parameter
        const uploadUrl = `${DRILL_BASE_URL}${API_ENDPOINTS.ANALYZE_DRILL}?drill_type=${encodeURIComponent(drillType)}`;
        console.log('Upload URL:', uploadUrl);
        console.log('Auth token present:', !!token);
        console.log('Auth token preview:', token ? token.substring(0, 20) + '...' : 'none');
        
        xhr.open('POST', uploadUrl);
        xhr.setRequestHeader('Authorization', `Bearer ${token}`);
        // Don't set Content-Type - let XMLHttpRequest set it automatically for FormData
        xhr.timeout = 120000; // 2 minutes
        
        console.log('Sending FormData...');
        xhr.send(formData);

      } catch (error) {
        reject(error);
      }
    });
  }

  // Intelligent retry with exponential backoff
  async uploadWithRetry(uploadFunction, maxRetries = 3) {
    let retries = 0;
    
    const attemptUpload = async () => {
      try {
        console.log(`Upload attempt ${retries + 1} of ${maxRetries + 1}`);
        return await uploadFunction();
      } catch (error) {
        console.error(`Upload attempt ${retries + 1} failed:`, error);
        console.error('Error message:', error.message);
        console.error('Error status:', error.status);
        console.error('Error details:', JSON.stringify(error, null, 2));
        
        retries++;
        
        // Check if we should retry
        if (retries <= maxRetries && this.isRetryableError(error)) {
          const delay = Math.pow(2, retries) * 1000; // Exponential backoff: 2s, 4s, 8s
          console.log(`Retrying upload in ${delay}ms...`);
          
          await new Promise(resolve => setTimeout(resolve, delay));
          return attemptUpload();
        }
        
        // Max retries reached or non-retryable error
        console.error('Upload failed after', retries, 'retries');
        return {
          success: false,
          error: error.message || error.toString()
        };
      }
    };
    
    return attemptUpload();
  }

  // Determine if an error should trigger a retry
  isRetryableError(error) {
    const message = error.message.toLowerCase();
    
    // Don't retry client errors (except timeouts)
    if (error.status >= 400 && error.status < 500) {
      return message.includes('timeout') || message.includes('network');
    }
    
    // Don't retry authentication errors
    if (message.includes('authentication') || message.includes('unauthorized')) {
      return false;
    }
    
    // Retry server errors and network issues
    return (
      message.includes('network') ||
      message.includes('timeout') ||
      message.includes('server error') ||
      error.status >= 500 ||
      !error.status // Network errors often don't have status
    );
  }

  async getAnalysisStatus(analysisId) {
    try {
      const response = await drillApi.get(API_ENDPOINTS.DRILL_STATUS(analysisId));
      return {
        success: true,
        status: response.data
      };
    } catch (error) {
      console.error('Failed to get analysis status:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to get analysis status'
      };
    }
  }

  async getAnalysisResults(analysisId) {
    try {
      const response = await drillApi.get(API_ENDPOINTS.DRILL_RESULTS(analysisId));
      return {
        success: true,
        results: response.data
      };
    } catch (error) {
      console.error('Failed to get analysis results:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to get results'
      };
    }
  }

  async logManualPractice(practiceData) {
    try {
      console.log('Logging manual practice:', practiceData);
      
      // Get user info
      const userStr = await AsyncStorage.getItem('user');
      const user = userStr ? JSON.parse(userStr) : null;
      
      const payload = {
        ...practiceData,
        user_id: user?.id || 'anonymous',
        timestamp: new Date().toISOString(),
      };
      
      // Real backend endpoint is now deployed and working!
      const response = await drillApi.post('/drill/manual-log', payload);
      return {
        success: true,
        message: response.data.message || 'Practice logged successfully',
        id: response.data.id
      };
    } catch (error) {
      console.error('Failed to log manual practice:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to log practice'
      };
    }
  }

  async getDrillProgress(drillType) {
    try {
      const response = await drillApi.get(`/api/drill-stats/${drillType}`);
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      console.error('Failed to get drill progress:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to load progress data'
      };
    }
  }

  async getUserStats() {
    try {
      // Get user info
      const userStr = await AsyncStorage.getItem('user');
      const user = userStr ? JSON.parse(userStr) : null;
      
      if (!user) {
        throw new Error('No user found');
      }

      // For now, calculate stats from drill results
      // TODO: Add dedicated /api/user/stats endpoint on server
      const response = await drillApi.get(`/api/user/stats`);
      return {
        success: true,
        stats: response.data
      };
    } catch (error) {
      console.error('Failed to get user stats:', error);
      // Return default stats with real user name
      const userStr = await AsyncStorage.getItem('user');
      const user = userStr ? JSON.parse(userStr) : null;
      const userName = user?.full_name?.split(' ')[0] || 'there';
      
      return {
        success: true,
        stats: {
          sessionsThisWeek: 0,
          currentStreak: 0,
          monthlyImprovement: 0,
          totalSessions: 0,
          userName: userName
        }
      };
    }
  }

  async getUserDrillProgress() {
    try {
      // Get all available drills
      const drillsResult = await this.getAvailableDrills();
      if (!drillsResult.success) {
        throw new Error('Failed to load drills');
      }

      // For each drill, get the user's progress
      const progressPromises = drillsResult.drills.map(async (drill) => {
        try {
          const progressResult = await this.getDrillProgress(drill.type);
          if (progressResult.success && progressResult.data) {
            return {
              ...drill,
              personalBest: progressResult.data.personal_best || 0,
              trend: progressResult.data.trend || 0,
              lastPracticed: progressResult.data.last_practiced || 'Never',
              improvementPercentage: progressResult.data.improvement_percentage || 0,
              totalSessions: progressResult.data.total_sessions || 0
            };
          } else {
            return {
              ...drill,
              personalBest: 0,
              trend: 0,
              lastPracticed: 'Never',
              improvementPercentage: 0,
              totalSessions: 0
            };
          }
        } catch (error) {
          console.error(`Failed to get progress for ${drill.type}:`, error);
          return {
            ...drill,
            personalBest: 0,
            trend: 0,
            lastPracticed: 'Never',
            improvementPercentage: 0,
            totalSessions: 0
          };
        }
      });

      const progressData = await Promise.all(progressPromises);
      
      return {
        success: true,
        progress: progressData
      };
    } catch (error) {
      console.error('Failed to get user drill progress:', error);
      return {
        success: false,
        error: error.message || 'Failed to load drill progress'
      };
    }
  }
}

export default new DrillService();