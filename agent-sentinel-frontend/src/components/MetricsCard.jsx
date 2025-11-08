// src/components/MetricsCard.jsx
import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { TrendingUp, TrendingDown } from '@mui/icons-material';

export const MetricsCard = ({ 
  title, 
  value, 
  subtitle, 
  icon: Icon, 
  color = '#3B82F6',
  trend = null 
}) => {
  return (
    <Card
      sx={{
        background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: 3,
        transition: 'all 0.3s ease',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: `0 8px 24px ${color}40`
        }
      }}
    >
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
          <Typography 
            variant="subtitle2" 
            sx={{ color: 'rgba(255, 255, 255, 0.6)', textTransform: 'uppercase', letterSpacing: 1 }}
          >
            {title}
          </Typography>
          <Icon sx={{ color, fontSize: 32 }} />
        </Box>

        <Typography 
          variant="h3" 
          sx={{ 
            color: 'white', 
            fontWeight: 700, 
            mb: 1,
            fontSize: { xs: '2rem', sm: '2.5rem' }
          }}
        >
          {value}
        </Typography>

        {subtitle && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {trend !== null && (
              trend > 0 ? (
                <TrendingUp sx={{ color: '#10B981', fontSize: 20 }} />
              ) : (
                <TrendingDown sx={{ color: '#EF4444', fontSize: 20 }} />
              )
            )}
            <Typography 
              variant="body2" 
              sx={{ color: 'rgba(255, 255, 255, 0.5)' }}
            >
              {subtitle}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default MetricsCard;