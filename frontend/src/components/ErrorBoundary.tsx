import React, { Component, ErrorInfo, ReactNode } from 'react'
import { AlertTriangle, RefreshCw, Home, Copy, Bug, ChevronDown, ChevronUp } from 'lucide-react'
import toast from 'react-hot-toast'

interface Props {
  children: ReactNode
  fallback?: ReactNode
  onError?: (error: Error, errorInfo: ErrorInfo) => void
}

interface State {
  hasError: boolean
  error?: Error
  errorInfo?: ErrorInfo
  showDetails: boolean
  errorId: string
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = {
      hasError: false,
      showDetails: false,
      errorId: ''
    }
  }

  static getDerivedStateFromError(error: Error): State {
    const errorId = `ERR_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    return {
      hasError: true,
      error,
      showDetails: false,
      errorId
    }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('🚨 ErrorBoundary caught an error:', error, errorInfo)

    this.setState({
      error,
      errorInfo
    })

    // Call optional error handler
    if (this.props.onError) {
      this.props.onError(error, errorInfo)
    }

    // Log comprehensive error details
    const errorDetails = {
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      errorId: this.state.errorId
    }

    console.error('📊 Error Details:', errorDetails)

    // Show toast notification
    toast.error(`Application Error: ${error.message}`, {
      duration: 6000,
      id: this.state.errorId
    })

    // Log to monitoring service
    if ((window as any).gtag) {
      (window as any).gtag('event', 'exception', {
        description: error.toString(),
        fatal: false
      })
    }
  }

  handleRetry = () => {
    this.setState({
      hasError: false,
      error: undefined,
      errorInfo: undefined,
      showDetails: false,
      errorId: ''
    })
  }

  handleGoHome = () => {
    window.location.href = '/'
  }

  handleRefresh = () => {
    window.location.reload()
  }

  toggleDetails = () => {
    this.setState(prev => ({ showDetails: !prev.showDetails }))
  }

  copyErrorDetails = () => {
    const errorDetails = {
      errorId: this.state.errorId,
      message: this.state.error?.message,
      stack: this.state.error?.stack,
      componentStack: this.state.errorInfo?.componentStack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href
    }

    navigator.clipboard.writeText(JSON.stringify(errorDetails, null, 2))
    toast.success('Error details copied to clipboard')
  }

  getErrorType = (error: Error): string => {
    if (error.message.includes('ChunkLoadError') || error.message.includes('Loading chunk')) {
      return 'Network/Loading Error'
    }
    if (error.message.includes('fetch') || error.message.includes('network')) {
      return 'Network Error'
    }
    if (error.message.includes('Cannot read property') || error.message.includes('undefined')) {
      return 'Data Error'
    }
    return 'Application Error'
  }

  getErrorSolution = (error: Error): string => {
    if (error.message.includes('ChunkLoadError') || error.message.includes('Loading chunk')) {
      return 'This usually happens after an app update. Please refresh the page.'
    }
    if (error.message.includes('fetch') || error.message.includes('network')) {
      return 'Check your internet connection and ensure the backend server is running.'
    }
    if (error.message.includes('Cannot read property') || error.message.includes('undefined')) {
      return 'This might be a temporary data issue. Try refreshing the page.'
    }
    return 'This is an unexpected error. Please try refreshing the page or contact support.'
  }

  render() {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback
      }

      // Default error UI
      return (
        <div className="min-h-[400px] flex items-center justify-center p-8">
          <div className="max-w-md w-full text-center">
            <div className="mb-6">
              <AlertTriangle className="h-16 w-16 text-red-500 mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-foreground mb-2">
                Something went wrong
              </h2>
              <p className="text-muted-foreground">
                An unexpected error occurred. Please try refreshing the page or contact support if the problem persists.
              </p>
            </div>

            {/* Error Details (Development Only) */}
            {import.meta.env.DEV && this.state.error && (
              <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-left">
                <h3 className="font-semibold text-red-800 dark:text-red-400 mb-2">
                  Error Details:
                </h3>
                <pre className="text-xs text-red-700 dark:text-red-300 overflow-auto">
                  {this.state.error.toString()}
                </pre>
                {this.state.errorInfo && (
                  <details className="mt-2">
                    <summary className="cursor-pointer text-red-600 dark:text-red-400 text-sm">
                      Component Stack
                    </summary>
                    <pre className="text-xs text-red-700 dark:text-red-300 mt-2 overflow-auto">
                      {this.state.errorInfo.componentStack}
                    </pre>
                  </details>
                )}
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-3 justify-center">
              <button
                onClick={this.handleRetry}
                className="btn-primary inline-flex items-center justify-center"
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Try Again
              </button>
              <button
                onClick={this.handleGoHome}
                className="btn-secondary inline-flex items-center justify-center"
              >
                <Home className="h-4 w-4 mr-2" />
                Go Home
              </button>
            </div>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary

// Canvas-specific error boundary
export const CanvasErrorBoundary: React.FC<{ children: ReactNode }> = ({ children }) => {
  return (
    <ErrorBoundary
      fallback={
        <div className="h-[800px] bg-muted/20 border-2 border-dashed border-red-300 rounded-lg flex items-center justify-center">
          <div className="text-center">
            <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-foreground mb-2">
              Canvas Error
            </h3>
            <p className="text-muted-foreground mb-4">
              The workflow canvas encountered an error. Please refresh the page.
            </p>
            <button
              onClick={() => window.location.reload()}
              className="btn-primary inline-flex items-center"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh Canvas
            </button>
          </div>
        </div>
      }
      onError={(error, errorInfo) => {
        console.error('Canvas Error:', error, errorInfo)
        // Send to monitoring service
      }}
    >
      {children}
    </ErrorBoundary>
  )
}
