// src/services/newsService.js
import axios from 'axios';
import { API_CONFIG } from '../config/constants';

class NewsService {
  constructor() {
    this.apiKey = API_CONFIG.NEWS_API_KEY;
    this.baseUrl = API_CONFIG.NEWS_API_URL;
  }

  /**
   * Fetch top headlines from NewsAPI
   */
  async fetchTopHeadlines(category = 'general', country = 'us') {
    try {
      const response = await axios.get(`${this.baseUrl}/top-headlines`, {
        params: {
          apiKey: this.apiKey,
          category,
          country,
          pageSize: 20
        }
      });

      return this.transformNewsData(response.data.articles);
    } catch (error) {
      console.error('Error fetching headlines:', error);
      throw new Error('Failed to fetch news');
    }
  }

  /**
   * Search news by keyword
   */
  async searchNews(query, sortBy = 'publishedAt') {
    try {
      const response = await axios.get(`${this.baseUrl}/everything`, {
        params: {
          apiKey: this.apiKey,
          q: query,
          sortBy,
          pageSize: 20,
          language: 'en'
        }
      });

      return this.transformNewsData(response.data.articles);
    } catch (error) {
      console.error('Error searching news:', error);
      throw new Error('Failed to search news');
    }
  }

  /**
   * Transform raw news data into our format
   */
  transformNewsData(articles) {
    return articles
      .filter(article => article.title && article.title !== '[Removed]')
      .map((article, index) => ({
        id: `news_${Date.now()}_${index}`,
        title: article.title,
        description: article.description || 'No description available',
        content: article.content || article.description,
        source: article.source.name,
        author: article.author || 'Unknown',
        url: article.url,
        imageUrl: article.urlToImage,
        publishedAt: new Date(article.publishedAt),
        
        // These will be analyzed by our AI (placeholder for now)
        analyzed: false,
        riskLevel: 'UNKNOWN',
        confidence: 0,
        verified: false,
        
        // Metadata
        category: this.categorizeNews(article.title + ' ' + article.description)
      }));
  }

  /**
   * Simple categorization based on keywords
   * (This is basic - AI will do better analysis later)
   */
  categorizeNews(text) {
    const lowerText = text.toLowerCase();
    
    if (this.hasKeywords(lowerText, ['health', 'vaccine', 'disease', 'medical', 'doctor', 'hospital'])) {
      return 'HEALTH';
    }
    if (this.hasKeywords(lowerText, ['election', 'vote', 'politics', 'government', 'president'])) {
      return 'POLITICS';
    }
    if (this.hasKeywords(lowerText, ['stock', 'market', 'economy', 'business', 'company'])) {
      return 'BUSINESS';
    }
    if (this.hasKeywords(lowerText, ['technology', 'tech', 'ai', 'software', 'computer'])) {
      return 'TECHNOLOGY';
    }
    
    return 'GENERAL';
  }

  hasKeywords(text, keywords) {
    return keywords.some(keyword => text.includes(keyword));
  }

  /**
   * Get news sources
   */
  async getSources(category = null) {
    try {
      const params = {
        apiKey: this.apiKey,
        language: 'en'
      };
      
      if (category) {
        params.category = category;
      }

      const response = await axios.get(`${this.baseUrl}/sources`, { params });
      return response.data.sources;
    } catch (error) {
      console.error('Error fetching sources:', error);
      throw new Error('Failed to fetch sources');
    }
  }
}

export const newsService = new NewsService();
export default newsService;