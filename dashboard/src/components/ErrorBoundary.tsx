import { Component, type ErrorInfo, type ReactNode } from 'react';
import { Alert, AlertTitle, Box, Button, Typography, Collapse } from '@mui/material';
import { Error as ErrorIcon, ExpandMore, ExpandLess } from '@mui/icons-material';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  componentName?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  showDetails: boolean;
}

/**
 * Error Boundary component for graceful error handling in dynamic rendering
 * 
 * Catches JavaScript errors in child components and displays a user-friendly
 * fallback UI instead of crashing the entire application.
 */
class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false
    };
  }

  static getDerivedStateFromError(_error: Error): Partial<State> {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log the error for debugging
    console.error(`Error Boundary caught an error in ${this.props.componentName || 'component'}:`, error, errorInfo);
    
    // Update state with error details
    this.setState({
      error,
      errorInfo
    });

    // Report error to monitoring service (if available)
    if (typeof (window as any).gtag === 'function') {
      (window as any).gtag('event', 'exception', {
        description: `Error in ${this.props.componentName || 'component'}: ${error.message}`,
        fatal: false
      });
    }
  }

  handleRetry = () => {
    // Reset error state to try rendering again
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false
    });
  };

  toggleDetails = () => {
    this.setState(prevState => ({
      showDetails: !prevState.showDetails
    }));
  };

  render() {
    if (this.state.hasError) {
      // Custom fallback UI provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default fallback UI
      return (
        <Box sx={{ p: 2 }}>
          <Alert severity="error" icon={<ErrorIcon />}>
            <AlertTitle>
              {this.props.componentName ? `Error in ${this.props.componentName}` : 'Rendering Error'}
            </AlertTitle>
            
            <Typography variant="body2" sx={{ mb: 2 }}>
              Something went wrong while displaying this content. This might be due to unexpected data format or a temporary issue.
            </Typography>

            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
              <Button
                variant="outlined"
                size="small"
                onClick={this.handleRetry}
                sx={{ minWidth: 'auto' }}
              >
                Try Again
              </Button>

              <Button
                variant="text"
                size="small"
                onClick={this.toggleDetails}
                endIcon={this.state.showDetails ? <ExpandLess /> : <ExpandMore />}
                sx={{ minWidth: 'auto' }}
              >
                {this.state.showDetails ? 'Hide' : 'Show'} Details
              </Button>
            </Box>

            <Collapse in={this.state.showDetails}>
              <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.100', borderRadius: 1 }}>
                <Typography variant="caption" color="text.secondary" component="div">
                  <strong>Error:</strong> {this.state.error?.message}
                </Typography>
                
                {this.state.error?.stack && (
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    component="pre"
                    sx={{
                      mt: 1,
                      fontSize: '0.7rem',
                      overflow: 'auto',
                      maxHeight: '200px',
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word'
                    }}
                  >
                    {this.state.error.stack}
                  </Typography>
                )}

                {this.state.errorInfo?.componentStack && (
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    component="pre"
                    sx={{
                      mt: 1,
                      fontSize: '0.7rem',
                      overflow: 'auto',
                      maxHeight: '200px',
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word'
                    }}
                  >
                    <strong>Component Stack:</strong>
                    {this.state.errorInfo.componentStack}
                  </Typography>
                )}
              </Box>
            </Collapse>
          </Alert>
        </Box>
      );
    }

    // No error, render children normally
    return this.props.children;
  }
}

export default ErrorBoundary;