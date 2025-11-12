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
  Skeleton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  LinearProgress
} from '@mui/material';
import {
  OpenInNew,
  AccessTime,
  Warning,
  Error as ErrorIcon,
  Info
} from '@mui/icons-material';
import { formatDistanceToNow, parseISO } from 'date-fns';

const getSeverityConfig = (severity) => {
  const configs = {
    CRITICAL: { color: '#DC2626', icon: ErrorIcon, label: '🔴 CRITICAL' },
    HIGH: { color: '#EA580C', icon: Warning, label: '🟠 HIGH' },
    MEDIUM: { color: '#F59E0B', icon: Warning, label: '🟡 MEDIUM' },
    LOW: { color: '#10B981', icon: Info, label: '🟢 LOW' }
  };
  return configs[severity?.toUpperCase()] || configs.MEDIUM;
};

export const NewsCard = ({ alert, loading = false, article = null }) => {
  const [openReport, setOpenReport] = React.useState(false);

  // SAFETY CHECK - if article is passed instead of alert, use it
  const data = article || alert;
  
  if (!data) {
    return (
      <Card sx={{ bgcolor: 'rgba(30, 41, 59, 0.8)' }}>
        <CardContent>
          <Typography color="error">No data available</Typography>
        </CardContent>
      </Card>
    );
  }

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

  const severity = data?.severity || data?.riskLevel || 'MEDIUM';
  const severityConfig = getSeverityConfig(severity);
  const SeverityIcon = severityConfig.icon;

  // Safe extraction of source name
  const sourceName = typeof data?.source === 'object' 
    ? data.source?.name || 'Unknown Source'
    : data.source || 'Unknown Source';

  // Safe date formatting
  let publishedTime = 'Unknown';
  if (data?.publishedAt) {
    try {
      const parsedDate = parseISO(data.publishedAt);
      if (!isNaN(parsedDate)) {
        publishedTime = formatDistanceToNow(parsedDate, { addSuffix: true });
      }
    } catch {
      publishedTime = 'Unknown';
    }
  }

  const falsePercent = Math.round((data?.falsehood_score || 0) * 100);
  const viralPercent = Math.round((data?.virality_score || 0) * 100);

  return (
    <>
      <Card
        sx={{
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          background: `linear-gradient(135deg, ${severityConfig.color}15 0%, ${severityConfig.color}05 100%)`,
          backdropFilter: 'blur(10px)',
          border: `2px solid ${severityConfig.color}`,
          borderRadius: 3,
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'translateY(-4px)',
            boxShadow: `0 12px 32px ${severityConfig.color}40`,
            borderColor: severityConfig.color
          }
        }}
      >
        <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
          {/* Header with Severity */}
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <SeverityIcon sx={{ fontSize: 20, color: severityConfig.color }} />
              <Chip
                label={severityConfig.label}
                size="small"
                sx={{
                  bgcolor: `${severityConfig.color}30`,
                  color: severityConfig.color,
                  fontWeight: 700,
                  borderRadius: 1
                }}
              />
            </Box>
            
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Chip
                label={`🔍 ${falsePercent}% Fake`}
                size="small"
                variant="outlined"
                sx={{ borderColor: severityConfig.color, color: severityConfig.color }}
              />
              <Chip
                label={`📈 ${viralPercent}% Viral`}
                size="small"
                variant="outlined"
                sx={{ borderColor: '#F59E0B', color: '#F59E0B' }}
              />
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
            {alert?.title || 'Untitled Alert'}
          </Typography>

          {/* Description/Summary */}
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
            {alert?.report_summary?.substring(0, 150) || 'No summary available'}
            {alert?.report_summary?.length > 150 ? '...' : ''}
          </Typography>

          {/* Score Visualization */}
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                Falsehood Score
              </Typography>
              <Typography variant="caption" sx={{ fontWeight: 600, color: severityConfig.color }}>
                {falsePercent}%
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={falsePercent}
              sx={{
                height: 6,
                borderRadius: 3,
                bgcolor: 'rgba(255, 255, 255, 0.1)',
                '& .MuiLinearProgress-bar': {
                  background: `linear-gradient(90deg, ${severityConfig.color}, ${severityConfig.color}dd)`,
                  borderRadius: 3
                }
              }}
            />
          </Box>

          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                Virality Score
              </Typography>
              <Typography variant="caption" sx={{ fontWeight: 600, color: '#F59E0B' }}>
                {viralPercent}%
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={viralPercent}
              sx={{
                height: 6,
                borderRadius: 3,
                bgcolor: 'rgba(255, 255, 255, 0.1)',
                '& .MuiLinearProgress-bar': {
                  background: 'linear-gradient(90deg, #F59E0B, #F59E0Bdd)',
                  borderRadius: 3
                }
              }}
            />
          </Box>

          <Divider sx={{ bgcolor: 'rgba(255, 255, 255, 0.1)', my: 2 }} />

          {/* Footer */}
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography
                variant="caption"
                sx={{ color: 'rgba(255, 255, 255, 0.5)', fontWeight: 600 }}
              >
                📡 {sourceName}
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <AccessTime sx={{ fontSize: 14, color: 'rgba(255, 255, 255, 0.4)' }} />
              <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.4)' }}>
                {publishedTime}
              </Typography>
            </Box>
          </Box>

          {/* Actions */}
          <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
            <Button
              variant="contained"
              size="small"
              onClick={() => setOpenReport(true)}
              sx={{
                flex: 1,
                bgcolor: severityConfig.color,
                '&:hover': { bgcolor: severityConfig.color, opacity: 0.9 }
              }}
            >
              Full Report
            </Button>
            <IconButton
              component="a"
              href={alert?.url}
              target="_blank"
              rel="noopener noreferrer"
              size="small"
              sx={{
                color: severityConfig.color,
                border: `1px solid ${severityConfig.color}`,
                '&:hover': { bgcolor: `${severityConfig.color}20` }
              }}
            >
              <OpenInNew fontSize="small" />
            </IconButton>
          </Box>
        </CardContent>
      </Card>

      {/* Full Report Dialog */}
      <Dialog open={openReport} onClose={() => setOpenReport(false)} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ fontWeight: 700 }}>
          🚨 Viral Misinformation Alert - Full Report
        </DialogTitle>
        <DialogContent dividers>
          <Box sx={{ whiteSpace: 'pre-wrap', fontSize: '0.875rem', fontFamily: 'monospace' }}>
            {alert?.report_summary || 'No detailed report available'}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenReport(false)}>Close</Button>
          {alert?.url && (
            <Button
              href={alert.url}
              target="_blank"
              rel="noopener noreferrer"
              variant="contained"
              endIcon={<OpenInNew />}
            >
              View Source
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </>
  );
};

export default NewsCard;