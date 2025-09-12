import Constants from 'expo-constants';

// API Configuration
// PRODUCTION: DigitalOcean Droplet
// DEVELOPMENT: Use your computer's local IP address
const IS_PRODUCTION = true; // Set to false for local development

export const API_BASE_URL = IS_PRODUCTION 
  ? 'https://soccertrainingapp.org' // Production DigitalOcean API
  : 'http://10.0.0.93:8000'; // Local development

// Supabase Configuration - From environment variables
export const SUPABASE_URL = Constants.expoConfig?.extra?.SUPABASE_URL || Constants.manifest?.extra?.SUPABASE_URL;
export const SUPABASE_ANON_KEY = Constants.expoConfig?.extra?.SUPABASE_ANON_KEY || Constants.manifest?.extra?.SUPABASE_ANON_KEY;

// Drill API Configuration (HTTPS enabled!)
export const DRILL_BASE_URL = Constants.expoConfig?.extra?.DRILL_BASE_URL || Constants.manifest?.extra?.DRILL_BASE_URL || 'https://soccertrainingapp.org';

// API Endpoints
export const API_ENDPOINTS = {
  // Auth (Production server)
  LOGIN: '/auth/login',
  REGISTER: '/auth/register',
  LOGOUT: '/auth/logout',
  ME: '/auth/me',
  REFRESH: '/auth/refresh',
  
  // Drills (Production server)
  AVAILABLE_DRILLS: '/drill/available',
  ANALYZE_DRILL: '/drill/analyze',
  DRILL_STATUS: (id) => `/drill/status/${id}`,
  DRILL_RESULTS: (id) => `/drill/results/${id}`,
};

// App Configuration
export const APP_CONFIG = {
  MAX_VIDEO_DURATION: 30, // 30 seconds for better user experience
  MIN_VIDEO_DURATION: 3,  // 3 seconds minimum
  MAX_FILE_SIZE: 100 * 1024 * 1024, // 100MB for longer videos
  POLLING_INTERVAL: 2000, // 2 seconds
};