/**
 * Processing status component to show real-time progress
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Alert,
  Stepper,
  Step,
  StepLabel,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  HourglassTop as HourglassIcon,
} from '@mui/icons-material';

import { ProcessingStatus as ProcessingStatusType } from '../types';
import WebSocketService from '../services/websocket';
import ResultDisplay from './ResultDisplay';

interface ProcessingStatusProps {
  alertId: string;
  onComplete?: () => void;
}

const ProcessingStatus: React.FC<ProcessingStatusProps> = ({ alertId, onComplete }) => {
  const [status, setStatus] = useState<ProcessingStatusType | null>(null);
  const [wsError, setWsError] = useState<string | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  
  const wsService = new WebSocketService();

  useEffect(() => {
    const initializeWebSocket = async () => {
      try {
        // Set up event handlers
        wsService.onStatusUpdateHandler((statusUpdate) => {
          setStatus(statusUpdate);
          
          // Call onComplete callback when processing is done
          if (statusUpdate.status === 'completed' && onComplete) {
            setTimeout(onComplete, 1000); // Small delay to show completion
          }
        });

        wsService.onErrorHandler((error) => {
          setWsError(error);
          setWsConnected(false);
        });

        wsService.onCloseHandler(() => {
          setWsConnected(false);
        });

        // Connect to WebSocket
        await wsService.connect(alertId);
        setWsConnected(true);
        
      } catch (error) {
        console.error('Failed to connect to WebSocket:', error);
        setWsError('Failed to connect to real-time updates');
      }
    };

    initializeWebSocket();

    // Cleanup on unmount
    return () => {
      wsService.disconnect();
    };
  }, [alertId, onComplete]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'queued':
        return 'info';
      case 'processing':
        return 'warning';
      case 'completed':
        return 'success';
      case 'error':
        return 'error';
      default:
        return 'primary';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon color="success" />;
      case 'error':
        return <ErrorIcon color="error" />;
      case 'processing':
      case 'queued':
        return <HourglassIcon color="primary" />;
      default:
        return null;
    }
  };

  const processingSteps = [
    'Alert received',
    'Downloading runbook',
    'Analyzing runbook and determining data requirements',
    'Gathering system data from MCP servers',
    'Performing AI analysis',
    'Analysis complete'
  ];

  const getCurrentStep = () => {
    if (!status) return 0;
    
    const currentStepText = status.current_step.toLowerCase();
    
    if (currentStepText.includes('alert received')) return 0;
    if (currentStepText.includes('downloading')) return 1;
    if (currentStepText.includes('analyzing') || currentStepText.includes('determining')) return 2;
    if (currentStepText.includes('gathering')) return 3;
    if (currentStepText.includes('performing')) return 4;
    if (currentStepText.includes('complete')) return 5;
    
    return Math.floor((status.progress / 100) * processingSteps.length);
  };

  if (!status) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Connecting to Alert Processing...
          </Typography>
          <LinearProgress />
          {wsError && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {wsError}
            </Alert>
          )}
        </CardContent>
      </Card>
    );
  }

  return (
    <Box>
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
            <Typography variant="h6">
              Alert Processing Status
            </Typography>
            <Box display="flex" alignItems="center" gap={1}>
              {getStatusIcon(status.status)}
              <Chip 
                label={status.status.toUpperCase()} 
                color={getStatusColor(status.status)} 
                size="small"
              />
            </Box>
          </Box>

          <Typography variant="body2" color="text.secondary" gutterBottom>
            Alert ID: {status.alert_id}
          </Typography>

          <Box mb={3}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
              <Typography variant="body2">
                {status.current_step}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {status.progress}%
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={status.progress} 
              color={getStatusColor(status.status)}
            />
          </Box>

          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle2">Processing Steps</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Stepper activeStep={getCurrentStep()} orientation="vertical">
                {processingSteps.map((step, index) => (
                  <Step key={step}>
                    <StepLabel>
                      <Typography variant="body2">{step}</Typography>
                    </StepLabel>
                  </Step>
                ))}
              </Stepper>
            </AccordionDetails>
          </Accordion>

          {status.error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              <Typography variant="body2">
                <strong>Error:</strong> {status.error}
              </Typography>
            </Alert>
          )}

          <Box mt={2}>
            <Typography variant="body2" color="text.secondary">
              Connection Status: {wsConnected ? '🟢 Connected' : '🔴 Disconnected'}
            </Typography>
          </Box>
        </CardContent>
      </Card>

      {status.result && status.status === 'completed' && (
        <Box mt={3}>
          <ResultDisplay result={status.result} />
        </Box>
      )}
    </Box>
  );
};

export default ProcessingStatus; 