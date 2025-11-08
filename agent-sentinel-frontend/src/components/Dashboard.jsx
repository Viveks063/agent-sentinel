// src/components/Dashboard.jsx
import React, { useState } from 'react';
import {
  Container,
  Grid,
  Box,
  Typography,
  CircularProgress,
  Alert,
  TextField,
  InputAdornment,
  Button,
  ButtonGroup,
  Paper
} from '@mui/material';
import {
  Newspaper,
  VerifiedUser,
  Warning,
  Search as SearchIcon
} from '@mui/icons-material';
import { Header } from './Header';
import { MetricsCard } from './MetricsCard';
import { NewsCard } from './NewsCard';
import { useNews } from '../hooks/useNews';

export const Dashboard = () => {
  const { news, loading, error, lastUpdated, refreshNews, searchNews } = useNews('general');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('ALL');

  const categories = ['ALL', 'HEALTH', 'POLITICS', 'BUSINESS', 'TECHNOLOGY'];

  // Calculate metrics
  const metrics = {
    total: news.length,
    verified: news.filter(n => n.verified).length,
    suspicious: news.filter(n => n.riskLevel === 'HIGH' || n.riskLevel === 'CRITICAL').length
  };

  const handleSearch = () => {
    if (searchQuery.trim()) {
      searchNews(searchQuery);
    }
  };

  const handleCategoryChange = (category) => {
    setSelectedCategory(category);
    // Filter logic would go here
  };

  const filteredNews = selectedCategory === 'ALL' 
    ? news 
    : news.filter(article => article.category === selectedCategory);

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#0f172a' }}>
      <Header 
        onRefresh={refreshNews} 
        lastUpdated={lastUpdated}
        isLive={true}
      />

      <Container maxWidth="xl" sx={{ py: 4 }}>
        {/* Metrics */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} sm={4}>
            <MetricsCard
              title="Total News"
              value={metrics.total}
              subtitle="articles monitored"
              icon={Newspaper}
              color="#3B82F6"
            />
          </Grid>
          <Grid item xs={12} sm={4}>
            <MetricsCard
              title="Verified"
              value={metrics.verified}
              subtitle="sources confirmed"
              icon={VerifiedUser}
              color="#10B981"
            />
          </Grid>
          <Grid item xs={12} sm={4}>
            <MetricsCard
              title="Suspicious"
              value={metrics.suspicious}
              subtitle="flagged for review"
              icon={Warning}
              color="#EF4444"
            />
          </Grid>
        </Grid>

        {/* Search & Filters */}
        <Paper
          elevation={0}
          sx={{
            p: 3,
            mb: 4,
            background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: 3
          }}
        >
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                placeholder="Search news..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon sx={{ color: 'rgba(255, 255, 255, 0.5)' }} />
                    </InputAdornment>
                  ),
                  sx: {
                    color: 'white',
                    bgcolor: 'rgba(15, 23, 42, 0.5)',
                    borderRadius: 2,
                    '& fieldset': { border: 'none' }
                  }
                }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', justifyContent: { xs: 'flex-start', md: 'flex-end' } }}>
                <ButtonGroup variant="outlined">
                  {categories.map((cat) => (
                    <Button
                      key={cat}
                      onClick={() => handleCategoryChange(cat)}
                      sx={{
                        color: selectedCategory === cat ? 'white' : 'rgba(255, 255, 255, 0.6)',
                        bgcolor: selectedCategory === cat ? 'rgba(59, 130, 246, 0.3)' : 'transparent',
                        borderColor: 'rgba(255, 255, 255, 0.2)',
                        '&:hover': {
                          bgcolor: 'rgba(59, 130, 246, 0.2)',
                          borderColor: 'rgba(255, 255, 255, 0.3)'
                        }
                      }}
                    >
                      {cat}
                    </Button>
                  ))}
                </ButtonGroup>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* Error Message */}
        {error && (
          <Alert severity="error" sx={{ mb: 4 }}>
            {error}
          </Alert>
        )}

        {/* Loading State */}
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
            <CircularProgress size={60} sx={{ color: '#3B82F6' }} />
          </Box>
        )}

        {/* News Grid */}
        {!loading && filteredNews.length > 0 && (
          <Grid container spacing={3}>
            {filteredNews.map((article) => (
              <Grid item xs={12} sm={6} lg={4} key={article.id}>
                <NewsCard article={article} />
              </Grid>
            ))}
          </Grid>
        )}

        {/* No News Found */}
        {!loading && filteredNews.length === 0 && (
          <Box
            sx={{
              textAlign: 'center',
              py: 8,
              px: 2
            }}
          >
            <Newspaper sx={{ fontSize: 80, color: 'rgba(255, 255, 255, 0.2)', mb: 2 }} />
            <Typography variant="h5" sx={{ color: 'rgba(255, 255, 255, 0.6)', mb: 1 }}>
              No news articles found
            </Typography>
            <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.4)' }}>
              Try adjusting your search or filters
            </Typography>
          </Box>
        )}
      </Container>
    </Box>
  );
};

export default Dashboard;