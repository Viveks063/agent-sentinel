import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Box,
  Typography,
  CircularProgress,
  Alert,
  Paper,
  Button,
  ButtonGroup,
  Card,
  CardContent,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Chip,
  IconButton,
  Divider
} from '@mui/material';
import {
  Newspaper,
  VerifiedUser,
  Warning,
  Refresh as RefreshIcon,
  OpenInNew,
  Error as ErrorIcon,
  AccessTime
} from '@mui/icons-material';

export const Dashboard = () => {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [openDialog, setOpenDialog] = useState(false);

  const fetchAlerts = async () => {
    try {
      setLoading(true);
      setError(null);

      const endpoint = "http://127.0.0.1:8000/alerts";
      
      console.log(`📡 Fetching from ${endpoint}...`);
      
      const response = await fetch(endpoint, {
        method: "GET",
        headers: { "Content-Type": "application/json" }
      });

      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`);
      }

      const data = await response.json();
      console.log("✅ Response:", data);

      const alerts = data.alerts || [];
      
      // CRITICAL FIX: Map API response to proper format
      const transformed = alerts.map((alert, idx) => {
        const sourceName = typeof alert.source === 'object' 
          ? alert.source?.name || 'Unknown'
          : alert.source || 'Unknown';

        return {
          id: alert.id || `alert_${idx}`,
          title: alert.title || "Untitled Alert",  // ✅ GET TITLE FROM ALERT
          description: alert.report_summary || alert.report || "No summary available",  // ✅ GET SUMMARY
          source: sourceName,
          url: alert.url || "#",
          publishedAt: alert.published_at || new Date().toISOString(),
          severity: alert.severity || "MEDIUM",
          falsehood_score: alert.falsehood_score || 0.33,
          virality_score: alert.virality_score || 0.33,
          full_report: alert.report || alert.report_summary || "No report available"
        };
      });

      console.log("📊 Transformed:", transformed);
      setNews(transformed);
      setLastUpdated(new Date());

    } catch (err) {
      console.error("❌ Error:", err);
      setError(err.message);
      setNews([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAlerts();
  }, []);

  const metrics = {
    total: news.length,
    critical: news.filter(n => n.severity === 'CRITICAL').length,
    suspicious: news.filter(n => ['CRITICAL', 'HIGH'].includes(n.severity)).length
  };

  const getSeverityColor = (severity) => {
    const colors = {
      CRITICAL: '#DC2626',
      HIGH: '#EA580C',
      MEDIUM: '#F59E0B',
      LOW: '#10B981'
    };
    return colors[severity] || '#F59E0B';
  };

  const getSeverityIcon = (severity) => {
    return severity === 'CRITICAL' ? <ErrorIcon /> : <Warning />;
  };

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#0f172a', py: 4 }}>
      <Container maxWidth="xl">
        
        {/* Header */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h3" sx={{ fontWeight: 700, mb: 1, color: 'white' }}>
            🚨 Agent Sentinel
          </Typography>
          <Typography variant="subtitle1" color="textSecondary">
            Viral Fake News Detection System - Live Monitoring
          </Typography>
        </Box>

        {/* Stats */}
        <Paper
          sx={{
            p: 3,
            mb: 4,
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: 'white',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}
        >
          <Box>
            <Typography variant="h4" sx={{ fontWeight: 700 }}>
              {news.length}
            </Typography>
            <Typography variant="body2">
              Active Alerts {lastUpdated && `• Last: ${lastUpdated.toLocaleTimeString()}`}
            </Typography>
          </Box>
          <Button
            variant="contained"
            startIcon={<RefreshIcon />}
            onClick={fetchAlerts}
            disabled={loading}
            sx={{
              bgcolor: 'rgba(255,255,255,0.2)',
              '&:hover': { bgcolor: 'rgba(255,255,255,0.3)' }
            }}
          >
            {loading ? <CircularProgress size={24} color="inherit" /> : 'Refresh'}
          </Button>
        </Paper>

        {/* Alert Banner */}
        {metrics.critical > 0 && (
          <Alert 
            severity="error" 
            sx={{ 
              mb: 4,
              bgcolor: 'rgba(239, 68, 68, 0.1)',
              borderColor: '#EF4444',
              color: 'white'
            }}
          >
            🚨 <strong>{metrics.critical} CRITICAL</strong> viral fake news alert(s) detected!
          </Alert>
        )}

        {/* Error State */}
        {error && (
          <Alert severity="error" sx={{ mb: 4 }}>
            ⚠️ {error}
          </Alert>
        )}

        {/* Loading State */}
        {loading && news.length === 0 && (
          <Box sx={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', py: 8 }}>
            <CircularProgress size={60} sx={{ color: '#3B82F6', mb: 2 }} />
            <Typography sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
              Scanning viral news for misinformation...
            </Typography>
          </Box>
        )}

        {/* Empty State */}
        {!loading && news.length === 0 && !error && (
          <Paper sx={{ p: 4, textAlign: 'center', bgcolor: 'rgba(16, 185, 129, 0.1)' }}>
            <VerifiedUser sx={{ fontSize: 80, color: '#10B981', mb: 2 }} />
            <Typography variant="h5" sx={{ color: 'white', mb: 1, fontWeight: 600 }}>
              ✅ No Alerts Found
            </Typography>
            <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
              No viral fake news detected. System is monitoring 9 sources...
            </Typography>
          </Paper>
        )}

        {/* Alerts Grid */}
        {news.length > 0 && (
          <>
            <Typography 
              variant="h6" 
              sx={{ 
                color: 'white', 
                mb: 3,
                display: 'flex',
                alignItems: 'center',
                gap: 1
              }}
            >
              🚨 Viral Fake News Alerts 
              <Typography 
                variant="caption" 
                sx={{ 
                  color: 'rgba(255, 255, 255, 0.5)',
                  ml: 'auto'
                }}
              >
                {news.length} alert{news.length !== 1 ? 's' : ''}
              </Typography>
            </Typography>

            <Grid container spacing={3}>
              {news.map((alert, idx) => (
                <Grid item xs={12} sm={6} lg={4} key={alert.id || idx}>
                  <Card
                    sx={{
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column',
                      background: `linear-gradient(135deg, ${getSeverityColor(alert.severity)}15 0%, ${getSeverityColor(alert.severity)}05 100%)`,
                      border: `2px solid ${getSeverityColor(alert.severity)}`,
                      borderRadius: 2,
                      transition: 'all 0.3s ease',
                      cursor: 'pointer',
                      '&:hover': {
                        transform: 'translateY(-4px)',
                        boxShadow: `0 12px 32px ${getSeverityColor(alert.severity)}40`
                      }
                    }}
                  >
                    <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                      {/* Header */}
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {getSeverityIcon(alert.severity)}
                          <Chip
                            label={alert.severity}
                            size="small"
                            sx={{
                              bgcolor: `${getSeverityColor(alert.severity)}30`,
                              color: getSeverityColor(alert.severity),
                              fontWeight: 700
                            }}
                          />
                        </Box>
                        
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <Chip
                            label={`${Math.round((alert.falsehood_score || 0) * 100)}% Fake`}
                            size="small"
                            variant="outlined"
                            sx={{ borderColor: getSeverityColor(alert.severity), color: getSeverityColor(alert.severity) }}
                          />
                          <Chip
                            label={`${Math.round((alert.virality_score || 0) * 100)}% Viral`}
                            size="small"
                            variant="outlined"
                            sx={{ borderColor: '#F59E0B', color: '#F59E0B' }}
                          />
                        </Box>
                      </Box>

                      {/* Title - NOW SHOWING ACTUAL FAKE NEWS */}
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
                          WebkitBoxOrient: 'vertical'
                        }}
                      >
                        {alert.title || 'Untitled Alert'}
                      </Typography>

                      {/* Description - NOW SHOWING SUMMARY */}
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
                        {alert.description || 'No summary available'}
                      </Typography>

                      {/* Scores */}
                      <Box sx={{ mb: 2 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                          <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                            Falsehood
                          </Typography>
                          <Typography variant="caption" sx={{ fontWeight: 600, color: getSeverityColor(alert.severity) }}>
                            {Math.round((alert.falsehood_score || 0) * 100)}%
                          </Typography>
                        </Box>
                        <LinearProgress
                          variant="determinate"
                          value={Math.round((alert.falsehood_score || 0) * 100)}
                          sx={{
                            height: 6,
                            borderRadius: 3,
                            bgcolor: 'rgba(255, 255, 255, 0.1)',
                            '& .MuiLinearProgress-bar': {
                              background: getSeverityColor(alert.severity),
                              borderRadius: 3
                            }
                          }}
                        />
                      </Box>

                      <Divider sx={{ bgcolor: 'rgba(255, 255, 255, 0.1)', my: 2 }} />

                      {/* Footer */}
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                        <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.5)' }}>
                          📡 {alert.source}
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <AccessTime sx={{ fontSize: 14, color: 'rgba(255, 255, 255, 0.4)' }} />
                          <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.4)' }}>
                            {new Date(alert.publishedAt).toLocaleTimeString()}
                          </Typography>
                        </Box>
                      </Box>

                      {/* Actions */}
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <Button
                          variant="contained"
                          size="small"
                          fullWidth
                          onClick={() => {
                            setSelectedAlert(alert);
                            setOpenDialog(true);
                          }}
                          sx={{
                            bgcolor: getSeverityColor(alert.severity),
                            '&:hover': { opacity: 0.9 }
                          }}
                        >
                          Full Report
                        </Button>
                        {alert.url && alert.url !== '#' && (
                          <IconButton
                            component="a"
                            href={alert.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            size="small"
                            sx={{
                              color: getSeverityColor(alert.severity),
                              border: `1px solid ${getSeverityColor(alert.severity)}`,
                              '&:hover': { bgcolor: `${getSeverityColor(alert.severity)}20` }
                            }}
                          >
                            <OpenInNew fontSize="small" />
                          </IconButton>
                        )}
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </>
        )}
      </Container>

      {/* Full Report Dialog */}
      <Dialog open={openDialog} onClose={() => setOpenDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ fontWeight: 700, bgcolor: '#0f172a', color: 'white' }}>
          🚨 Full Misinformation Report
        </DialogTitle>
        <DialogContent dividers sx={{ bgcolor: '#0f172a', color: 'white' }}>
          <Box sx={{ whiteSpace: 'pre-wrap', fontSize: '0.875rem', fontFamily: 'monospace', maxHeight: '60vh', overflow: 'auto' }}>
            {selectedAlert?.full_report || 'No report available'}
          </Box>
        </DialogContent>
        <DialogActions sx={{ bgcolor: '#0f172a' }}>
          <Button onClick={() => setOpenDialog(false)}>Close</Button>
          {selectedAlert?.url && selectedAlert.url !== '#' && (
            <Button
              href={selectedAlert.url}
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
    </Box>
  );
};

export default Dashboard;