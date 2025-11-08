// src/components/Header.jsx
import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
  Chip,
  Container
} from '@mui/material';
import {
  Shield,
  Refresh,
  Search
} from '@mui/icons-material';

export const Header = ({ onRefresh, lastUpdated, isLive }) => {
  const formatTime = (date) => {
    if (!date) return 'Never';
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <AppBar 
      position="sticky" 
      elevation={0}
      sx={{
        background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
      }}
    >
      <Container maxWidth="xl">
        <Toolbar sx={{ py: 1 }}>
          {/* Logo & Title */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexGrow: 1 }}>
            <Shield sx={{ fontSize: 40, color: '#EF4444' }} />
            <Box>
              <Typography 
                variant="h5" 
                component="div" 
                sx={{ 
                  fontWeight: 700,
                  background: 'linear-gradient(135deg, #EF4444 0%, #DC2626 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent'
                }}
              >
                Agent Sentinel
              </Typography>
              <Typography 
                variant="caption" 
                sx={{ color: 'rgba(255, 255, 255, 0.6)' }}
              >
                AI-Powered News Verification
              </Typography>
            </Box>
          </Box>

          {/* Status Indicators */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {/* Live Status */}
            <Chip
              icon={
                <Box
                  sx={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    bgcolor: isLive ? '#10B981' : '#6B7280',
                    animation: isLive ? 'pulse 2s infinite' : 'none',
                    '@keyframes pulse': {
                      '0%, 100%': { opacity: 1 },
                      '50%': { opacity: 0.5 }
                    }
                  }}
                />
              }
              label={isLive ? 'LIVE' : 'OFFLINE'}
              size="small"
              sx={{
                bgcolor: isLive ? 'rgba(16, 185, 129, 0.2)' : 'rgba(107, 114, 128, 0.2)',
                color: isLive ? '#10B981' : '#9CA3AF',
                fontWeight: 600
              }}
            />

            {/* Last Updated */}
            <Typography 
              variant="caption" 
              sx={{ 
                color: 'rgba(255, 255, 255, 0.5)',
                display: { xs: 'none', sm: 'block' }
              }}
            >
              Updated: {formatTime(lastUpdated)}
            </Typography>

            {/* Refresh Button */}
            <IconButton 
              onClick={onRefresh}
              sx={{ 
                color: 'white',
                '&:hover': {
                  bgcolor: 'rgba(255, 255, 255, 0.1)',
                  transform: 'rotate(180deg)',
                  transition: 'transform 0.3s ease'
                }
              }}
            >
              <Refresh />
            </IconButton>
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
};

export default Header;