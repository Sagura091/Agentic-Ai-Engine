import React, { useState, useEffect, useRef } from 'react'
import { logger, LogLevel, LogEntry } from '../utils/logger'

interface LogViewerProps {
  isOpen: boolean
  onClose: () => void
}

const LogViewer: React.FC<LogViewerProps> = ({ isOpen, onClose }) => {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [filteredLogs, setFilteredLogs] = useState<LogEntry[]>([])
  const [selectedLevel, setSelectedLevel] = useState<LogLevel | 'ALL'>('ALL')
  const [searchTerm, setSearchTerm] = useState('')
  const [autoScroll, setAutoScroll] = useState(true)
  const [stats, setStats] = useState(logger.getStats())
  const logsEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isOpen) {
      updateLogs()
      const interval = setInterval(updateLogs, 1000) // Update every second
      return () => clearInterval(interval)
    }
  }, [isOpen])

  useEffect(() => {
    filterLogs()
  }, [logs, selectedLevel, searchTerm])

  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [filteredLogs, autoScroll])

  const updateLogs = () => {
    const currentLogs = logger.getLogs()
    setLogs(currentLogs)
    setStats(logger.getStats())
  }

  const filterLogs = () => {
    let filtered = logs

    // Filter by level
    if (selectedLevel !== 'ALL') {
      filtered = filtered.filter(log => log.level === selectedLevel)
    }

    // Filter by search term
    if (searchTerm) {
      const term = searchTerm.toLowerCase()
      filtered = filtered.filter(log =>
        log.message.toLowerCase().includes(term) ||
        log.source.toLowerCase().includes(term) ||
        (log.data && JSON.stringify(log.data).toLowerCase().includes(term))
      )
    }

    setFilteredLogs(filtered)
  }

  const handleDownloadLogs = () => {
    logger.downloadLogs()
  }

  const handleClearLogs = () => {
    if (confirm('Are you sure you want to clear all logs?')) {
      logger.clearLogs()
      updateLogs()
    }
  }

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString()
  }

  const getLevelColor = (level: LogLevel) => {
    switch (level) {
      case LogLevel.DEBUG:
        return 'text-gray-500'
      case LogLevel.INFO:
        return 'text-blue-600'
      case LogLevel.WARN:
        return 'text-yellow-600'
      case LogLevel.ERROR:
        return 'text-red-600'
      case LogLevel.FATAL:
        return 'text-red-800 font-bold'
      default:
        return 'text-gray-600'
    }
  }

  const getLevelBadgeColor = (level: LogLevel) => {
    switch (level) {
      case LogLevel.DEBUG:
        return 'bg-gray-100 text-gray-800'
      case LogLevel.INFO:
        return 'bg-blue-100 text-blue-800'
      case LogLevel.WARN:
        return 'bg-yellow-100 text-yellow-800'
      case LogLevel.ERROR:
        return 'bg-red-100 text-red-800'
      case LogLevel.FATAL:
        return 'bg-red-200 text-red-900'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center">
      <div className="bg-white rounded-lg shadow-xl w-11/12 h-5/6 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <div className="flex items-center space-x-4">
            <h2 className="text-xl font-semibold">Frontend Logs</h2>
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <span>Session: {stats.sessionId.slice(-8)}</span>
              <span>•</span>
              <span>Total: {stats.totalLogs}</span>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={handleDownloadLogs}
              className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
            >
              Download
            </button>
            <button
              onClick={handleClearLogs}
              className="px-3 py-1 bg-red-500 text-white rounded text-sm hover:bg-red-600"
            >
              Clear
            </button>
            <button
              onClick={onClose}
              className="px-3 py-1 bg-gray-500 text-white rounded text-sm hover:bg-gray-600"
            >
              Close
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="flex items-center space-x-4 p-4 border-b bg-gray-50">
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium">Level:</label>
            <select
              value={selectedLevel}
              onChange={(e) => setSelectedLevel(e.target.value as LogLevel | 'ALL')}
              className="border rounded px-2 py-1 text-sm"
            >
              <option value="ALL">All</option>
              <option value={LogLevel.DEBUG}>Debug</option>
              <option value={LogLevel.INFO}>Info</option>
              <option value={LogLevel.WARN}>Warning</option>
              <option value={LogLevel.ERROR}>Error</option>
              <option value={LogLevel.FATAL}>Fatal</option>
            </select>
          </div>

          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium">Search:</label>
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search logs..."
              className="border rounded px-2 py-1 text-sm w-64"
            />
          </div>

          <div className="flex items-center space-x-2">
            <label className="flex items-center space-x-1 text-sm">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
              />
              <span>Auto-scroll</span>
            </label>
          </div>

          <div className="flex-1"></div>

          <div className="text-sm text-gray-600">
            Showing {filteredLogs.length} of {logs.length} logs
          </div>
        </div>

        {/* Stats */}
        <div className="flex items-center space-x-4 p-2 bg-gray-100 text-sm">
          {Object.entries(stats.logsByLevel).map(([level, count]) => (
            <div key={level} className="flex items-center space-x-1">
              <span className={`px-2 py-1 rounded text-xs ${getLevelBadgeColor(level as LogLevel)}`}>
                {level}
              </span>
              <span className="text-gray-600">{count}</span>
            </div>
          ))}
        </div>

        {/* Logs */}
        <div className="flex-1 overflow-auto p-4 bg-gray-50">
          <div className="space-y-1 font-mono text-sm">
            {filteredLogs.map((log, index) => (
              <div
                key={index}
                className="bg-white border rounded p-2 hover:bg-gray-50"
              >
                <div className="flex items-start space-x-2">
                  <span className="text-gray-500 text-xs whitespace-nowrap">
                    {formatTimestamp(log.timestamp)}
                  </span>
                  <span className={`px-2 py-1 rounded text-xs whitespace-nowrap ${getLevelBadgeColor(log.level)}`}>
                    {log.level}
                  </span>
                  <span className="text-purple-600 text-xs whitespace-nowrap">
                    [{log.source}]
                  </span>
                  <span className={`flex-1 ${getLevelColor(log.level)}`}>
                    {log.message}
                  </span>
                </div>
                {log.data && (
                  <div className="mt-1 ml-20 text-xs text-gray-600 bg-gray-100 p-2 rounded">
                    <pre className="whitespace-pre-wrap">{JSON.stringify(log.data, null, 2)}</pre>
                  </div>
                )}
                {log.stackTrace && (
                  <div className="mt-1 ml-20 text-xs text-red-600 bg-red-50 p-2 rounded">
                    <pre className="whitespace-pre-wrap">{log.stackTrace}</pre>
                  </div>
                )}
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </div>
      </div>
    </div>
  )
}

export default LogViewer
