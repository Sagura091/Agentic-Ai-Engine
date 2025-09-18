import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react'
import toast from 'react-hot-toast'

export interface AppError {
  id: string
  type: 'network' | 'validation' | 'authentication' | 'server' | 'client' | 'unknown'
  message: string
  details?: string
  timestamp: Date
  source?: string
  stack?: string
  retryable: boolean
  severity: 'low' | 'medium' | 'high' | 'critical'
}

interface ErrorContextType {
  errors: AppError[]
  reportError: (error: Error | string, context?: Partial<AppError>) => string
  clearError: (errorId: string) => void
  clearAllErrors: () => void
  getErrorsByType: (type: AppError['type']) => AppError[]
  getErrorsBySeverity: (severity: AppError['severity']) => AppError[]
  retryableErrors: AppError[]
  criticalErrors: AppError[]
}

const ErrorContext = createContext<ErrorContextType | undefined>(undefined)

export const useError = () => {
  const context = useContext(ErrorContext)
  if (context === undefined) {
    throw new Error('useError must be used within an ErrorProvider')
  }
  return context
}

interface ErrorProviderProps {
  children: ReactNode
  maxErrors?: number
}

export const ErrorProvider: React.FC<ErrorProviderProps> = ({ 
  children, 
  maxErrors = 100 
}) => {
  const [errors, setErrors] = useState<AppError[]>([])

  const generateErrorId = () => `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`

  const determineErrorType = (error: Error | string): AppError['type'] => {
    const message = typeof error === 'string' ? error : error.message
    
    if (message.includes('fetch') || message.includes('network') || message.includes('NetworkError')) {
      return 'network'
    }
    if (message.includes('401') || message.includes('unauthorized') || message.includes('authentication')) {
      return 'authentication'
    }
    if (message.includes('400') || message.includes('validation') || message.includes('invalid')) {
      return 'validation'
    }
    if (message.includes('500') || message.includes('server') || message.includes('internal')) {
      return 'server'
    }
    if (message.includes('ChunkLoadError') || message.includes('Loading chunk')) {
      return 'client'
    }
    
    return 'unknown'
  }

  const determineSeverity = (type: AppError['type'], message: string): AppError['severity'] => {
    if (type === 'authentication' || message.includes('critical') || message.includes('fatal')) {
      return 'critical'
    }
    if (type === 'server' || type === 'network') {
      return 'high'
    }
    if (type === 'validation' || type === 'client') {
      return 'medium'
    }
    return 'low'
  }

  const isRetryable = (type: AppError['type']): boolean => {
    return ['network', 'server', 'client'].includes(type)
  }

  const reportError = useCallback((
    error: Error | string, 
    context: Partial<AppError> = {}
  ): string => {
    const errorId = generateErrorId()
    const message = typeof error === 'string' ? error : error.message
    const type = context.type || determineErrorType(error)
    const severity = context.severity || determineSeverity(type, message)
    
    const appError: AppError = {
      id: errorId,
      type,
      message,
      details: context.details || (typeof error === 'object' ? error.stack : undefined),
      timestamp: new Date(),
      source: context.source,
      stack: typeof error === 'object' ? error.stack : undefined,
      retryable: context.retryable ?? isRetryable(type),
      severity,
      ...context
    }

    setErrors(prev => {
      const newErrors = [appError, ...prev].slice(0, maxErrors)
      return newErrors
    })

    // Show toast notification based on severity
    const toastMessage = `${appError.source ? `[${appError.source}] ` : ''}${message}`
    
    switch (severity) {
      case 'critical':
        toast.error(toastMessage, { 
          duration: 8000,
          id: errorId,
          icon: '🚨'
        })
        break
      case 'high':
        toast.error(toastMessage, { 
          duration: 6000,
          id: errorId 
        })
        break
      case 'medium':
        toast.error(toastMessage, { 
          duration: 4000,
          id: errorId 
        })
        break
      case 'low':
        toast(toastMessage, { 
          duration: 3000,
          id: errorId,
          icon: '⚠️'
        })
        break
    }

    // Log to console for debugging
    console.error('🚨 Error reported:', appError)

    // Send to monitoring service if available
    if ((window as any).gtag) {
      (window as any).gtag('event', 'exception', {
        description: message,
        fatal: severity === 'critical',
        error_type: type,
        error_source: context.source
      })
    }

    return errorId
  }, [maxErrors])

  const clearError = useCallback((errorId: string) => {
    setErrors(prev => prev.filter(error => error.id !== errorId))
    toast.dismiss(errorId)
  }, [])

  const clearAllErrors = useCallback(() => {
    setErrors([])
    toast.dismiss()
  }, [])

  const getErrorsByType = useCallback((type: AppError['type']) => {
    return errors.filter(error => error.type === type)
  }, [errors])

  const getErrorsBySeverity = useCallback((severity: AppError['severity']) => {
    return errors.filter(error => error.severity === severity)
  }, [errors])

  const retryableErrors = errors.filter(error => error.retryable)
  const criticalErrors = errors.filter(error => error.severity === 'critical')

  const value: ErrorContextType = {
    errors,
    reportError,
    clearError,
    clearAllErrors,
    getErrorsByType,
    getErrorsBySeverity,
    retryableErrors,
    criticalErrors
  }

  return (
    <ErrorContext.Provider value={value}>
      {children}
    </ErrorContext.Provider>
  )
}

// Hook for handling async operations with error reporting
export const useAsyncError = () => {
  const { reportError } = useError()

  const executeWithErrorHandling = useCallback(async (
    operation: () => Promise<any>,
    context?: Partial<AppError>
  ): Promise<any> => {
    try {
      return await operation()
    } catch (error) {
      reportError(error as Error, context)
      return null
    }
  }, [reportError])

  return { executeWithErrorHandling }
}

// Hook for API calls with automatic error handling
export const useApiError = () => {
  const { reportError } = useError()

  const handleApiError = useCallback((
    error: any,
    endpoint?: string,
    method?: string
  ) => {
    const context: Partial<AppError> = {
      source: `API ${method || 'call'}${endpoint ? ` ${endpoint}` : ''}`,
      type: 'network'
    }

    if (error.response) {
      // Server responded with error status
      context.type = error.response.status >= 500 ? 'server' : 'validation'
      context.details = `Status: ${error.response.status}, Data: ${JSON.stringify(error.response.data)}`
    } else if (error.request) {
      // Request was made but no response received
      context.type = 'network'
      context.details = 'No response received from server'
    } else {
      // Something else happened
      context.type = 'client'
      context.details = error.message
    }

    return reportError(error.message || 'API call failed', context)
  }, [reportError])

  return { handleApiError }
}

export default ErrorProvider
