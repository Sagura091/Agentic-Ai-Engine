/**
 * Frontend Logging System
 * Captures all console logs, errors, warnings, API calls, and user interactions
 * Stores logs locally and optionally sends to backend
 */

export enum LogLevel {
  DEBUG = 'DEBUG',
  INFO = 'INFO',
  WARN = 'WARN',
  ERROR = 'ERROR',
  FATAL = 'FATAL'
}

export interface LogEntry {
  timestamp: string
  level: LogLevel
  message: string
  data?: any
  source: string
  userAgent: string
  url: string
  userId?: string
  sessionId: string
  stackTrace?: string
}

export interface LoggerConfig {
  maxLogEntries: number
  enableConsoleCapture: boolean
  enableApiLogging: boolean
  enableErrorCapture: boolean
  enableUserInteractionLogging: boolean
  logToBackend: boolean
  backendEndpoint?: string
  logLevels: LogLevel[]
  excludePatterns: string[]
}

class FrontendLogger {
  private logs: LogEntry[] = []
  private sessionId: string
  private config: LoggerConfig
  private originalConsole: {
    log: typeof console.log
    warn: typeof console.warn
    error: typeof console.error
    info: typeof console.info
    debug: typeof console.debug
  }

  constructor(config: Partial<LoggerConfig> = {}) {
    this.sessionId = this.generateSessionId()
    this.config = {
      maxLogEntries: 1000,
      enableConsoleCapture: true,
      enableApiLogging: true,
      enableErrorCapture: true,
      enableUserInteractionLogging: true,
      logToBackend: false,
      logLevels: [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARN, LogLevel.ERROR, LogLevel.FATAL],
      excludePatterns: [],
      ...config
    }

    // Store original console methods
    this.originalConsole = {
      log: console.log.bind(console),
      warn: console.warn.bind(console),
      error: console.error.bind(console),
      info: console.info.bind(console),
      debug: console.debug.bind(console)
    }

    this.init()
  }

  private init(): void {
    if (this.config.enableConsoleCapture) {
      this.interceptConsole()
    }

    if (this.config.enableErrorCapture) {
      this.setupErrorCapture()
    }

    if (this.config.enableUserInteractionLogging) {
      this.setupUserInteractionLogging()
    }

    if (this.config.enableApiLogging) {
      this.setupApiLogging()
    }

    // Log system initialization
    this.log(LogLevel.INFO, 'Frontend logging system initialized', {
      sessionId: this.sessionId,
      config: this.config
    }, 'Logger')
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  private shouldLog(level: LogLevel, message: string): boolean {
    // Check if log level is enabled
    if (!this.config.logLevels.includes(level)) {
      return false
    }

    // Check exclude patterns
    return !this.config.excludePatterns.some(pattern => 
      message.toLowerCase().includes(pattern.toLowerCase())
    )
  }

  private createLogEntry(
    level: LogLevel,
    message: string,
    data?: any,
    source: string = 'Unknown',
    stackTrace?: string
  ): LogEntry {
    return {
      timestamp: new Date().toISOString(),
      level,
      message,
      data,
      source,
      userAgent: navigator.userAgent,
      url: window.location.href,
      sessionId: this.sessionId,
      stackTrace
    }
  }

  public log(
    level: LogLevel,
    message: string,
    data?: any,
    source: string = 'App',
    stackTrace?: string
  ): void {
    if (!this.shouldLog(level, message)) {
      return
    }

    const logEntry = this.createLogEntry(level, message, data, source, stackTrace)
    
    // Add to logs array
    this.logs.push(logEntry)

    // Maintain max log entries
    if (this.logs.length > this.config.maxLogEntries) {
      this.logs = this.logs.slice(-this.config.maxLogEntries)
    }

    // Store in localStorage
    this.saveToLocalStorage()

    // Send to backend if enabled
    if (this.config.logToBackend && this.config.backendEndpoint) {
      this.sendToBackend(logEntry)
    }

    // Also log to original console for development
    this.logToOriginalConsole(level, message, data)
  }

  private logToOriginalConsole(level: LogLevel, message: string, data?: any): void {
    const logMessage = `[${level}] ${message}`
    
    switch (level) {
      case LogLevel.DEBUG:
        this.originalConsole.debug(logMessage, data)
        break
      case LogLevel.INFO:
        this.originalConsole.info(logMessage, data)
        break
      case LogLevel.WARN:
        this.originalConsole.warn(logMessage, data)
        break
      case LogLevel.ERROR:
      case LogLevel.FATAL:
        this.originalConsole.error(logMessage, data)
        break
      default:
        this.originalConsole.log(logMessage, data)
    }
  }

  private interceptConsole(): void {
    // Intercept console.log
    console.log = (...args: any[]) => {
      this.log(LogLevel.INFO, args.join(' '), args.length > 1 ? args : undefined, 'Console')
      this.originalConsole.log(...args)
    }

    // Intercept console.warn
    console.warn = (...args: any[]) => {
      this.log(LogLevel.WARN, args.join(' '), args.length > 1 ? args : undefined, 'Console')
      this.originalConsole.warn(...args)
    }

    // Intercept console.error
    console.error = (...args: any[]) => {
      this.log(LogLevel.ERROR, args.join(' '), args.length > 1 ? args : undefined, 'Console')
      this.originalConsole.error(...args)
    }

    // Intercept console.info
    console.info = (...args: any[]) => {
      this.log(LogLevel.INFO, args.join(' '), args.length > 1 ? args : undefined, 'Console')
      this.originalConsole.info(...args)
    }

    // Intercept console.debug
    console.debug = (...args: any[]) => {
      this.log(LogLevel.DEBUG, args.join(' '), args.length > 1 ? args : undefined, 'Console')
      this.originalConsole.debug(...args)
    }
  }

  private setupErrorCapture(): void {
    // Capture unhandled errors
    window.addEventListener('error', (event) => {
      this.log(
        LogLevel.ERROR,
        `Unhandled Error: ${event.message}`,
        {
          filename: event.filename,
          lineno: event.lineno,
          colno: event.colno,
          error: event.error
        },
        'ErrorHandler',
        event.error?.stack
      )
    })

    // Capture unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      this.log(
        LogLevel.ERROR,
        `Unhandled Promise Rejection: ${event.reason}`,
        {
          reason: event.reason,
          promise: event.promise
        },
        'PromiseHandler',
        event.reason?.stack
      )
    })
  }

  private setupUserInteractionLogging(): void {
    // Log page navigation
    const originalPushState = history.pushState
    const originalReplaceState = history.replaceState

    history.pushState = function(...args) {
      logger.log(LogLevel.INFO, `Navigation: ${args[2]}`, { state: args[0] }, 'Navigation')
      return originalPushState.apply(history, args)
    }

    history.replaceState = function(...args) {
      logger.log(LogLevel.INFO, `Navigation Replace: ${args[2]}`, { state: args[0] }, 'Navigation')
      return originalReplaceState.apply(history, args)
    }

    // Log clicks on important elements
    document.addEventListener('click', (event) => {
      const target = event.target as HTMLElement
      if (target.tagName === 'BUTTON' || target.closest('button') || target.getAttribute('role') === 'button') {
        this.log(
          LogLevel.DEBUG,
          `Button Click: ${target.textContent?.trim() || target.className}`,
          {
            element: target.tagName,
            className: target.className,
            id: target.id
          },
          'UserInteraction'
        )
      }
    })
  }

  private setupApiLogging(): void {
    // Intercept fetch requests
    const originalFetch = window.fetch
    window.fetch = async (...args: Parameters<typeof fetch>) => {
      const startTime = Date.now()
      const url = args[0].toString()
      const options = args[1] || {}

      this.log(
        LogLevel.DEBUG,
        `API Request: ${options.method || 'GET'} ${url}`,
        {
          url,
          method: options.method || 'GET',
          headers: options.headers,
          body: options.body
        },
        'API'
      )

      try {
        const response = await originalFetch(...args)
        const duration = Date.now() - startTime

        this.log(
          response.ok ? LogLevel.INFO : LogLevel.WARN,
          `API Response: ${response.status} ${url}`,
          {
            url,
            status: response.status,
            statusText: response.statusText,
            duration: `${duration}ms`,
            headers: Object.fromEntries(response.headers.entries())
          },
          'API'
        )

        return response
      } catch (error) {
        const duration = Date.now() - startTime
        this.log(
          LogLevel.ERROR,
          `API Error: ${url}`,
          {
            url,
            error: error instanceof Error ? error.message : String(error),
            duration: `${duration}ms`
          },
          'API',
          error instanceof Error ? error.stack : undefined
        )
        throw error
      }
    }

    // Intercept XMLHttpRequest
    const originalXHROpen = XMLHttpRequest.prototype.open
    const originalXHRSend = XMLHttpRequest.prototype.send

    XMLHttpRequest.prototype.open = function(method: string, url: string | URL, ...args: any[]) {
      (this as any)._loggerMethod = method;
      (this as any)._loggerUrl = url.toString();
      (this as any)._loggerStartTime = Date.now()
      return originalXHROpen.call(this, method, url, ...args)
    }

    XMLHttpRequest.prototype.send = function(body?: any) {
      logger.log(
        LogLevel.DEBUG,
        `XHR Request: ${(this as any)._loggerMethod} ${(this as any)._loggerUrl}`,
        { method: (this as any)._loggerMethod, url: (this as any)._loggerUrl, body },
        'XHR'
      )

      this.addEventListener('loadend', () => {
        const duration = Date.now() - (this as any)._loggerStartTime
        logger.log(
          this.status >= 200 && this.status < 300 ? LogLevel.INFO : LogLevel.WARN,
          `XHR Response: ${this.status} ${(this as any)._loggerUrl}`,
          {
            url: (this as any)._loggerUrl,
            status: this.status,
            statusText: this.statusText,
            duration: `${duration}ms`,
            responseType: this.responseType
          },
          'XHR'
        )
      })

      return originalXHRSend.call(this, body)
    }
  }

  private saveToLocalStorage(): void {
    try {
      const logsToSave = this.logs.slice(-100) // Save last 100 logs to localStorage
      localStorage.setItem('frontend_logs', JSON.stringify(logsToSave))
      localStorage.setItem('frontend_logs_session', this.sessionId)
    } catch (error) {
      // localStorage might be full or unavailable
      this.originalConsole.warn('Failed to save logs to localStorage:', error)
    }
  }

  private async sendToBackend(logEntry: LogEntry): Promise<void> {
    if (!this.config.backendEndpoint) return

    try {
      await fetch(this.config.backendEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(logEntry)
      })
    } catch (error) {
      // Don't log backend errors to avoid infinite loops
      this.originalConsole.warn('Failed to send log to backend:', error)
    }
  }

  // Public methods for manual logging
  public debug(message: string, data?: any, source?: string): void {
    this.log(LogLevel.DEBUG, message, data, source)
  }

  public info(message: string, data?: any, source?: string): void {
    this.log(LogLevel.INFO, message, data, source)
  }

  public warn(message: string, data?: any, source?: string): void {
    this.log(LogLevel.WARN, message, data, source)
  }

  public error(message: string, data?: any, source?: string, error?: Error): void {
    this.log(LogLevel.ERROR, message, data, source, error?.stack)
  }

  public fatal(message: string, data?: any, source?: string, error?: Error): void {
    this.log(LogLevel.FATAL, message, data, source, error?.stack)
  }

  // Utility methods
  public getLogs(level?: LogLevel): LogEntry[] {
    if (level) {
      return this.logs.filter(log => log.level === level)
    }
    return [...this.logs]
  }

  public getLogsAsText(): string {
    return this.logs.map(log =>
      `[${log.timestamp}] [${log.level}] [${log.source}] ${log.message}${
        log.data ? ` | Data: ${JSON.stringify(log.data)}` : ''
      }${log.stackTrace ? ` | Stack: ${log.stackTrace}` : ''}`
    ).join('\n')
  }

  public downloadLogs(filename?: string): void {
    const logsText = this.getLogsAsText()
    const blob = new Blob([logsText], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)

    const a = document.createElement('a')
    a.href = url
    a.download = filename || `frontend_logs_${this.sessionId}_${new Date().toISOString().split('T')[0]}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)

    this.info('Logs downloaded', { filename: a.download }, 'Logger')
  }

  public clearLogs(): void {
    this.logs = []
    localStorage.removeItem('frontend_logs')
    this.info('Logs cleared', undefined, 'Logger')
  }

  public getSessionId(): string {
    return this.sessionId
  }

  public updateConfig(newConfig: Partial<LoggerConfig>): void {
    this.config = { ...this.config, ...newConfig }
    this.info('Logger configuration updated', { config: this.config }, 'Logger')
  }

  public getStats(): {
    totalLogs: number
    logsByLevel: Record<LogLevel, number>
    sessionId: string
    oldestLog?: string
    newestLog?: string
  } {
    const logsByLevel = this.logs.reduce((acc, log) => {
      acc[log.level] = (acc[log.level] || 0) + 1
      return acc
    }, {} as Record<LogLevel, number>)

    return {
      totalLogs: this.logs.length,
      logsByLevel,
      sessionId: this.sessionId,
      oldestLog: this.logs[0]?.timestamp,
      newestLog: this.logs[this.logs.length - 1]?.timestamp
    }
  }
}

// Create global logger instance
export const logger = new FrontendLogger({
  enableConsoleCapture: true,
  enableApiLogging: true,
  enableErrorCapture: true,
  enableUserInteractionLogging: true,
  logToBackend: false, // Set to true if you want to send logs to backend
  backendEndpoint: 'http://localhost:8888/api/v1/logs', // Backend logging endpoint
  maxLogEntries: 1000,
  logLevels: [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARN, LogLevel.ERROR, LogLevel.FATAL],
  excludePatterns: ['socket.io', 'heartbeat'] // Exclude noisy logs
})

// Export types and logger
export default logger

// Add global logger to window for debugging
declare global {
  interface Window {
    frontendLogger: FrontendLogger
  }
}

window.frontendLogger = logger
