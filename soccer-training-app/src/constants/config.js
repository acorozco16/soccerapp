// API Configuration
// For development: Use your computer's local IP address
// To find your IP on Mac: System Preferences > Network > WiFi > Advanced > TCP/IP
// Change this to your computer's IP address (e.g., http://192.168.1.100:8000)
export const API_BASE_URL = 'http://10.0.0.93:8000';

// API Endpoints
export const API_ENDPOINTS = {
  // Auth
  LOGIN: '/auth/login',
  REGISTER: '/auth/register',
  LOGOUT: '/auth/logout',
  ME: '/auth/me',
  
  // Drills
  AVAILABLE_DRILLS: '/drill/available',
  ANALYZE_DRILL: '/drill/analyze',
  DRILL_STATUS: (id) => `/drill/status/${id}`,
  DRILL_RESULTS: (id) => `/drill/results/${id}`,
};

// App Configuration
export const APP_CONFIG = {
  MAX_VIDEO_DURATION: 300, // 5 minutes in seconds
  MIN_VIDEO_DURATION: 10,  // 10 seconds
  MAX_FILE_SIZE: 100 * 1024 * 1024, // 100MB
  POLLING_INTERVAL: 2000, // 2 seconds
};