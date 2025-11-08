// src/config/constants.js

export const API_CONFIG = {
  NEWS_API_KEY: import.meta.env.VITE_NEWS_API_KEY || 'YOUR_NEWS_API_KEY_HERE',
  NEWS_API_URL: 'https://newsapi.org/v2',
  BACKEND_URL: import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000',
  WEBSOCKET_URL: import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws'
};

export const NEWS_CATEGORIES = {
  ALL: 'general',
  HEALTH: 'health',
  TECHNOLOGY: 'technology',
  POLITICS: 'politics',
  BUSINESS: 'business',
  SCIENCE: 'science'
};

export const RISK_LEVELS = {
  LOW: { color: '#10B981', label: 'Low Risk' },
  MEDIUM: { color: '#F59E0B', label: 'Medium Risk' },
  HIGH: { color: '#EF4444', label: 'High Risk' },
  CRITICAL: { color: '#DC2626', label: 'Critical' }
};

export const REFRESH_INTERVAL = 300000; // 60 seconds