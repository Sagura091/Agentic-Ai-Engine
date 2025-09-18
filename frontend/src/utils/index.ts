// Utility functions for the Agentic AI Frontend

import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

/**
 * Combine class names with Tailwind CSS merge
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Format bytes to human readable string
 */
export function formatBytes(bytes: number, decimals = 2): string {
  if (bytes === 0) return '0 Bytes'

  const k = 1024
  const dm = decimals < 0 ? 0 : decimals
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']

  const i = Math.floor(Math.log(bytes) / Math.log(k))

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i]
}

/**
 * Format duration in seconds to human readable string
 */
export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`
  } else if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}m ${remainingSeconds.toFixed(0)}s`
  } else {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    return `${hours}h ${minutes}m`
  }
}

/**
 * Debounce function calls
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: number | null = null
  
  return (...args: Parameters<T>) => {
    if (timeout) {
      clearTimeout(timeout)
    }
    
    timeout = setTimeout(() => {
      func(...args)
    }, wait)
  }
}

/**
 * Throttle function calls
 */
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean = false
  
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args)
      inThrottle = true
      setTimeout(() => (inThrottle = false), limit)
    }
  }
}

/**
 * Generate a random ID
 */
export function generateId(prefix = 'id'): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
}

/**
 * Generate a UUID v4
 */
export function generateUUID(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }

  // Fallback for older browsers
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0
    const v = c === 'x' ? r : (r & 0x3 | 0x8)
    return v.toString(16)
  })
}

/**
 * Request Animation Frame throttle
 */
export function rafThrottle<T extends (...args: any[]) => any>(
  func: T
): ((...args: Parameters<T>) => void) {
  let rafId: number | null = null

  return (...args: Parameters<T>) => {
    if (rafId) {
      cancelAnimationFrame(rafId)
    }

    rafId = requestAnimationFrame(() => {
      func(...args)
    })
  }
}

/**
 * Transform point coordinates
 */
export function transformPoint(
  point: { x: number; y: number },
  transform: { x: number; y: number; scale: number }
): { x: number; y: number } {
  return {
    x: (point.x - transform.x) / transform.scale,
    y: (point.y - transform.y) / transform.scale
  }
}

/**
 * Inverse transform point coordinates
 */
export function inverseTransformPoint(
  point: { x: number; y: number },
  transform: { x: number; y: number; scale: number }
): { x: number; y: number } {
  return {
    x: point.x * transform.scale + transform.x,
    y: point.y * transform.scale + transform.y
  }
}

/**
 * Constrain point to bounds
 */
export function constrainToBounds(
  point: { x: number; y: number },
  bounds: { x: number; y: number; width: number; height: number },
  nodeSize: { width: number; height: number }
): { x: number; y: number } {
  return {
    x: Math.max(bounds.x, Math.min(bounds.x + bounds.width - nodeSize.width, point.x)),
    y: Math.max(bounds.y, Math.min(bounds.y + bounds.height - nodeSize.height, point.y))
  }
}

/**
 * Calculate distance between two points
 */
export function calculateDistance(
  point1: { x: number; y: number },
  point2: { x: number; y: number }
): number {
  const dx = point2.x - point1.x
  const dy = point2.y - point1.y
  return Math.sqrt(dx * dx + dy * dy)
}

/**
 * Snap point to grid
 */
export function snapToGrid(
  point: { x: number; y: number },
  gridSize: number
): { x: number; y: number } {
  return {
    x: Math.round(point.x / gridSize) * gridSize,
    y: Math.round(point.y / gridSize) * gridSize
  }
}

/**
 * Capitalize first letter of a string
 */
export function capitalize(str: string): string {
  return str.charAt(0).toUpperCase() + str.slice(1)
}

/**
 * Truncate text to specified length
 */
export function truncate(text: string, length: number): string {
  if (text.length <= length) return text
  return text.slice(0, length) + '...'
}

/**
 * Check if a value is empty (null, undefined, empty string, empty array, empty object)
 */
export function isEmpty(value: any): boolean {
  if (value == null) return true
  if (typeof value === 'string') return value.trim() === ''
  if (Array.isArray(value)) return value.length === 0
  if (typeof value === 'object') return Object.keys(value).length === 0
  return false
}

/**
 * Deep clone an object
 */
export function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== 'object') return obj
  if (obj instanceof Date) return new Date(obj.getTime()) as any
  if (obj instanceof Array) return obj.map(item => deepClone(item)) as any
  if (typeof obj === 'object') {
    const clonedObj = {} as any
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        clonedObj[key] = deepClone(obj[key])
      }
    }
    return clonedObj
  }
  return obj
}

/**
 * Get status color based on status string
 */
export function getStatusColor(status: string): string {
  switch (status.toLowerCase()) {
    case 'active':
    case 'healthy':
    case 'success':
    case 'completed':
    case 'online':
      return 'text-green-600 bg-green-100 dark:bg-green-900/20 dark:text-green-400'
    case 'warning':
    case 'pending':
    case 'executing':
    case 'running':
      return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20 dark:text-yellow-400'
    case 'error':
    case 'failed':
    case 'unhealthy':
    case 'offline':
    case 'inactive':
      return 'text-red-600 bg-red-100 dark:bg-red-900/20 dark:text-red-400'
    case 'idle':
    case 'paused':
    case 'stopped':
      return 'text-gray-600 bg-gray-100 dark:bg-gray-900/20 dark:text-gray-400'
    default:
      return 'text-blue-600 bg-blue-100 dark:bg-blue-900/20 dark:text-blue-400'
  }
}

/**
 * Format number with commas
 */
export function formatNumber(num: number): string {
  return num.toLocaleString()
}

/**
 * Calculate percentage
 */
export function calculatePercentage(value: number, total: number): number {
  if (total === 0) return 0
  return Math.round((value / total) * 100)
}

/**
 * Validate email format
 */
export function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return emailRegex.test(email)
}

/**
 * Validate URL format
 */
export function isValidUrl(url: string): boolean {
  try {
    new URL(url)
    return true
  } catch {
    return false
  }
}

/**
 * Get relative time string
 */
export function getRelativeTime(date: Date): string {
  const now = new Date()
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000)

  if (diffInSeconds < 60) {
    return 'just now'
  } else if (diffInSeconds < 3600) {
    const minutes = Math.floor(diffInSeconds / 60)
    return `${minutes} minute${minutes > 1 ? 's' : ''} ago`
  } else if (diffInSeconds < 86400) {
    const hours = Math.floor(diffInSeconds / 3600)
    return `${hours} hour${hours > 1 ? 's' : ''} ago`
  } else {
    const days = Math.floor(diffInSeconds / 86400)
    return `${days} day${days > 1 ? 's' : ''} ago`
  }
}

/**
 * Copy text to clipboard
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch (error) {
    // Fallback for older browsers
    const textArea = document.createElement('textarea')
    textArea.value = text
    document.body.appendChild(textArea)
    textArea.focus()
    textArea.select()
    try {
      document.execCommand('copy')
      document.body.removeChild(textArea)
      return true
    } catch (fallbackError) {
      document.body.removeChild(textArea)
      return false
    }
  }
}

/**
 * Download data as file
 */
export function downloadAsFile(data: string, filename: string, type = 'text/plain'): void {
  const blob = new Blob([data], { type })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

/**
 * Parse JSON safely
 */
export function safeJsonParse<T>(json: string, fallback: T): T {
  try {
    return JSON.parse(json)
  } catch {
    return fallback
  }
}

/**
 * Local storage helpers
 */
export const storage = {
  get: <T>(key: string, fallback: T): T => {
    try {
      const item = localStorage.getItem(key)
      return item ? JSON.parse(item) : fallback
    } catch {
      return fallback
    }
  },
  
  set: (key: string, value: any): void => {
    try {
      localStorage.setItem(key, JSON.stringify(value))
    } catch (error) {
      console.warn('Failed to save to localStorage:', error)
    }
  },
  
  remove: (key: string): void => {
    try {
      localStorage.removeItem(key)
    } catch (error) {
      console.warn('Failed to remove from localStorage:', error)
    }
  },
  
  clear: (): void => {
    try {
      localStorage.clear()
    } catch (error) {
      console.warn('Failed to clear localStorage:', error)
    }
  }
}
