import { useState, useEffect } from 'react';
import { Box, LinearProgress, CircularProgress, Typography } from '@mui/material';
import { formatDurationMs } from '../utils/timestamp';

interface ProgressIndicatorProps {
  status: 'completed' | 'failed' | 'in_progress' | 'pending';
  startedAt?: number; // Unix timestamp in microseconds
  duration?: number | null; // Duration in milliseconds
  variant?: 'linear' | 'circular';
  showDuration?: boolean;
  size?: 'small' | 'medium' | 'large';
}

/**
 * ProgressIndicator component - Phase 5
 * Shows progress indicators for active sessions and duration for completed ones
 */
function ProgressIndicator({ 
  status, 
  startedAt, 
  duration, 
  variant = 'linear',
  showDuration = true,
  size = 'medium'
}: ProgressIndicatorProps) {
  // State for live duration updates
  const [liveDuration, setLiveDuration] = useState<number | null>(null);

  // Calculate live duration for active sessions
  const getLiveDuration = () => {
    if (duration) return duration; // Use final duration if available
    if (startedAt && (status === 'in_progress' || status === 'pending')) {
      const now = Date.now() * 1000; // Convert to microseconds
      return Math.max(0, (now - startedAt) / 1000); // Convert to milliseconds
    }
    return null;
  };

  // Live ticking timer for active sessions
  useEffect(() => {
    // Only start timer for active sessions without final duration
    if ((status === 'in_progress' || status === 'pending') && startedAt && !duration) {
      // Update immediately
      setLiveDuration(getLiveDuration());
      
      // Then update every second
      const timer = setInterval(() => {
        const newDuration = getLiveDuration();
        setLiveDuration(newDuration);
      }, 1000);

      return () => clearInterval(timer);
    } else {
      // For completed sessions or when duration is provided, use that instead
      setLiveDuration(duration ?? null);
    }
  }, [status, startedAt, duration]);

  // Use live duration for display, fallback to calculated duration
  const currentDuration = liveDuration ?? getLiveDuration();
  const sizeMap = {
    small: 16,
    medium: 24,
    large: 32
  };

  // For active sessions, show progress indicator
  if (status === 'in_progress') {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        {variant === 'circular' ? (
          <CircularProgress 
            size={sizeMap[size]} 
            color="info"
            variant="indeterminate"
          />
        ) : (
          <LinearProgress 
            variant="indeterminate" 
            sx={{ 
              flexGrow: 1, 
              height: size === 'small' ? 6 : 8,
              borderRadius: 1,
              '& .MuiLinearProgress-bar': {
                borderRadius: 1,
              }
            }} 
            color="info"
          />
        )}
        {showDuration && currentDuration && (
          <Typography variant="caption" color="text.secondary">
            {formatDurationMs(currentDuration)}
          </Typography>
        )}
      </Box>
    );
  }

  // For pending sessions, show waiting indicator
  if (status === 'pending') {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        {variant === 'circular' ? (
          <CircularProgress 
            size={sizeMap[size]} 
            color="warning"
            variant="indeterminate"
          />
        ) : (
          <LinearProgress 
            variant="indeterminate" 
            sx={{ 
              flexGrow: 1, 
              height: size === 'small' ? 6 : 8,
              borderRadius: 1,
              '& .MuiLinearProgress-bar': {
                borderRadius: 1,
              }
            }} 
            color="warning"
          />
        )}
        {showDuration && currentDuration && (
          <Typography variant="caption" color="text.secondary">
            {formatDurationMs(currentDuration)}
          </Typography>
        )}
      </Box>
    );
  }

  // For completed/failed sessions, show duration if available
  if (showDuration && currentDuration) {
    return (
      <Typography 
        variant="caption" 
        color={status === 'completed' ? 'success.main' : status === 'failed' ? 'error.main' : 'text.secondary'}
        sx={{ fontWeight: 500 }}
      >
        {formatDurationMs(currentDuration)}
      </Typography>
    );
  }

  return null;
}

export default ProgressIndicator; 