// API Configuration
// PRODUCTION: DigitalOcean Droplet
// DEVELOPMENT: Use your computer's local IP address
const IS_PRODUCTION = true; // Set to false for local development

export const API_BASE_URL = IS_PRODUCTION 
  ? 'http://147.182.224.87:8000' // DigitalOcean Droplet
  : 'http://10.0.0.93:8000'; // Local development

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