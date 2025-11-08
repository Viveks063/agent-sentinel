// src/components/NewsCard.jsx
import React from 'react';
import {
  Card,
  CardContent,
  CardMedia,
  Typography,
  Box,
  Chip,
  IconButton,
  Divider,
  Skeleton
} from '@mui/material';
import {
  OpenInNew,
  AccessTime,
  Verified,
  Warning,
  CheckCircle
} from '@mui/icons-material';
import { formatDistanceToNow } from 'date-fns';

const getCategoryColor = (category) => {
  const colors = {
    HEALTH: '#3B82F6',
    POLITICS: '#10B981',
    BUSINESS: '#F59E0B',
    TECHNOLOGY: '#8B5CF6',
    GENERAL: '#6B7280'
  };
  return colors[category] || colors.GENERAL;
};

const getRiskLevelConfig = (level) => {
  const configs = {
    LOW: { color: '#10B981', icon: CheckCircle, label: 'Low Risk' },
    MEDIUM: { color: '#F59E0B', icon: Warning, label: 'Medium Risk' },
    HIGH: { color: '#EF4444', icon: Warning, label: 'High Risk' },
    CRITICAL: { color: '#DC2626', icon: Warning, label: 'Critical' },
    UNKNOWN: { color: '#6B7280', icon: AccessTime, label: 'Analyzing...' }
  };
  return configs[level] || configs.UNKNOWN;
};

export const NewsCard = ({ article, loading = false }) => {
  if (loading) {
    return (
      <Card sx={{ height: '100%', bgcolor: 'rgba(30, 41, 59, 0.8)' }}>
        <Skeleton variant="rectangular" height={200} />
        <CardContent>
          <Skeleton variant="text" height={32} />
          <Skeleton variant="text" height={24} />
          <Skeleton variant="text" height={24} width="60%" />
        </CardContent>
      </Card>
    );
  }

  const riskConfig = getRiskLevelConfig(article.riskLevel);
  const RiskIcon = riskConfig.icon;

  return (
    <Card
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: 3,
        transition: 'all 0.3s ease',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: '0 12px 32px rgba(0, 0, 0, 0.4)',
          borderColor: getCategoryColor(article.category)
        }
      }}
    >
      {/* Image */}
      {article.imageUrl && (
        <CardMedia
          component="img"
          height="200"
          image={article.imageUrl}
          alt={article.title}
          sx={{
            objectFit: 'cover',
            borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
          }}
          onError={(e) => {
            e.target.style.display = 'none';
          }}
        />
      )}

      <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
        {/* Header: Category & Risk Level */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Chip
            label={article.category}
            size="small"
            sx={{
              bgcolor: `${getCategoryColor(article.category)}20`,
              color: getCategoryColor(article.category),
              fontWeight: 600,
              borderRadius: 2
            }}
          />
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <RiskIcon sx={{ fontSize: 18, color: riskConfig.color }} />
            <Typography 
              variant="caption" 
              sx={{ color: riskConfig.color, fontWeight: 600 }}
            >
              {riskConfig.label}
            </Typography>
          </Box>
        </Box>

        {/* Title */}
        <Typography 
          variant="h6" 
          sx={{ 
            color: 'white', 
            fontWeight: 600, 
            mb: 1.5,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            lineHeight: 1.4
          }}
        >
          {article.title}
        </Typography>

        {/* Description */}
        <Typography 
          variant="body2" 
          sx={{ 
            color: 'rgba(255, 255, 255, 0.7)', 
            mb: 2,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            display: '-webkit-box',
            WebkitLineClamp: 3,
            WebkitBoxOrient: 'vertical',
            flexGrow: 1
          }}
        >
          {article.description}
        </Typography>

        <Divider sx={{ bgcolor: 'rgba(255, 255, 255, 0.1)', my: 2 }} />

        {/* Footer: Source & Time */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography 
              variant="caption" 
              sx={{ 
                color: 'rgba(255, 255, 255, 0.5)',
                fontWeight: 600
              }}
            >
              {article.source}
            </Typography>
            {article.verified && (
              <Verified sx={{ fontSize: 16, color: '#10B981' }} />
            )}
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AccessTime sx={{ fontSize: 14, color: 'rgba(255, 255, 255, 0.4)' }} />
            <Typography 
              variant="caption" 
              sx={{ color: 'rgba(255, 255, 255, 0.4)' }}
            >
              {formatDistanceToNow(article.publishedAt, { addSuffix: true })}
            </Typography>
          </Box>
        </Box>

        {/* Read More Button */}
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
          <IconButton
            component="a"
            href={article.url}
            target="_blank"
            rel="noopener noreferrer"
            size="small"
            sx={{
              color: getCategoryColor(article.category),
              bgcolor: `${getCategoryColor(article.category)}20`,
              '&:hover': {
                bgcolor: `${getCategoryColor(article.category)}30`
              }
            }}
          >
            <OpenInNew fontSize="small" />
          </IconButton>
        </Box>
      </CardContent>
    </Card>
  );
};

export default NewsCard;