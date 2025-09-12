// API Configuration
// PRODUCTION: DigitalOcean Droplet
// DEVELOPMENT: Use your computer's local IP address
const IS_PRODUCTION = true; // Set to false for local development

export const API_BASE_URL = IS_PRODUCTION 
  ? 'https://nxumfeldylzpqwqlvszz.supabase.co/functions/v1' // Supabase Edge Functions
  : 'http://10.0.0.93:8000'; // Local development

// Supabase Configuration
export const SUPABASE_URL = 'https://nxumfeldylzpqwqlvszz.supabase.co';
export const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im54dW1mZWxkeWx6cHF3cWx2c3p6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM5MTY1NDksImV4cCI6MjA2OTQ5MjU0OX0.D2WvA9Ld2YalWbum6qi5CBvXxmj75v1BuDb-NKrJkxo';

// Drill API Configuration (HTTPS enabled!)
export const DRILL_BASE_URL = 'https://soccertrainingapp.org';

// API Endpoints
export const API_ENDPOINTS = {
  // Auth (Supabase Edge Functions)
  LOGIN: '/auth-login',
  REGISTER: '/auth-register',
  LOGOUT: '/auth-logout',
  ME: '/auth-me',
  REFRESH: '/auth-refresh',
  
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