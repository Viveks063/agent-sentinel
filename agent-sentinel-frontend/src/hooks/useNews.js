// src/hooks/useNews.js
import { useState, useEffect, useCallback } from 'react';
import { newsService } from '../services/newsService';
import { REFRESH_INTERVAL } from '../config/constants';

export const useNews = (category = 'general', autoRefresh = true) => {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const fetchNews = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const articles = await newsService.fetchTopHeadlines(category);
      setNews(articles);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err.message);
      console.error('Error fetching news:', err);
    } finally {
      setLoading(false);
    }
  }, [category]);

  const searchNews = useCallback(async (query) => {
    try {
      setLoading(true);
      setError(null);
      
      const articles = await newsService.searchNews(query);
      setNews(articles);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err.message);
      console.error('Error searching news:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshNews = useCallback(() => {
    fetchNews();
  }, [fetchNews]);

  // Initial fetch
  useEffect(() => {
    fetchNews();
  }, [fetchNews]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      fetchNews();
    }, REFRESH_INTERVAL);

    return () => clearInterval(interval);
  }, [fetchNews, autoRefresh]);

  return {
    news,
    loading,
    error,
    lastUpdated,
    refreshNews,
    searchNews
  };
};

export default useNews;