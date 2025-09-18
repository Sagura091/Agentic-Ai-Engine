import React, { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  Bot,
  GitBranch,
  MessageSquare,
  Activity,
  Settings,
  Menu,
  X,
  Sun,
  Moon,
  Monitor,
  Brain,
  Zap,
  Bug,
  Database,
  Search,
  Upload,
  BookOpen
} from 'lucide-react'
import { useTheme } from '../../contexts/ThemeContext'
import { useSocket } from '../../contexts/SocketContext'
import { useError } from '../../contexts/ErrorContext'
import DebugPanel from '../DebugPanel'

interface LayoutProps {
  children: React.ReactNode
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [debugPanelOpen, setDebugPanelOpen] = useState(false)
  const { theme, setTheme } = useTheme()
  const { isConnected } = useSocket()
  const { errors } = useError()
  const location = useLocation()

  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
    { name: 'Conversational Agents', href: '/conversational-agents', icon: Brain },
    { name: 'Agent Builder', href: '/agents', icon: Bot },
    { name: 'Agent Tasks', href: '/agent-tasks', icon: Zap },
    { name: 'RAG Management', href: '/rag', icon: Database },
    { name: 'Knowledge Bases', href: '/knowledge-bases', icon: BookOpen },
    { name: 'Workflow Designer', href: '/workflows', icon: GitBranch },
    { name: 'Agent Chat', href: '/chat', icon: MessageSquare },
    { name: 'Monitoring', href: '/monitoring', icon: Activity },
    { name: 'Settings', href: '/settings', icon: Settings },
  ]

  const isCurrentPath = (path: string) => {
    return location.pathname === path || location.pathname.startsWith(path + '/')
  }

  const toggleTheme = () => {
    if (theme === 'light') {
      setTheme('dark')
    } else if (theme === 'dark') {
      setTheme('system')
    } else {
      setTheme('light')
    }
  }

  const getThemeIcon = () => {
    switch (theme) {
      case 'light':
        return <Sun className="h-4 w-4" />
      case 'dark':
        return <Moon className="h-4 w-4" />
      default:
        return <Monitor className="h-4 w-4" />
    }
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Mobile sidebar backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div
        className={`fixed inset-y-0 left-0 z-50 w-64 bg-card border-r border-border transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex h-full flex-col">
          {/* Logo */}
          <div className="flex h-16 items-center justify-between px-6 border-b border-border">
            <div className="flex items-center space-x-2">
              <div className="h-8 w-8 bg-primary rounded-lg flex items-center justify-center">
                <Bot className="h-5 w-5 text-primary-foreground" />
              </div>
              <span className="text-lg font-semibold text-foreground">
                Agentic AI
              </span>
            </div>
            <button
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden p-1 rounded-md hover:bg-accent"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Connection status */}
          <div className="px-6 py-3 border-b border-border">
            <div className="flex items-center space-x-2">
              <div
                className={`h-2 w-2 rounded-full ${
                  isConnected ? 'bg-green-500' : 'bg-red-500'
                }`}
              />
              <span className="text-sm text-muted-foreground">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-4 space-y-1">
            {navigation.map((item) => {
              const Icon = item.icon
              const current = isCurrentPath(item.href)
              
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`group flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                    current
                      ? 'bg-primary text-primary-foreground'
                      : 'text-muted-foreground hover:text-foreground hover:bg-accent'
                  }`}
                  onClick={() => setSidebarOpen(false)}
                >
                  <Icon
                    className={`mr-3 h-5 w-5 flex-shrink-0 ${
                      current ? 'text-primary-foreground' : 'text-muted-foreground group-hover:text-foreground'
                    }`}
                  />
                  {item.name}
                </Link>
              )
            })}
          </nav>

          {/* Theme toggle */}
          <div className="px-4 py-4 border-t border-border">
            <button
              onClick={toggleTheme}
              className="flex items-center w-full px-3 py-2 text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-accent rounded-md transition-colors"
            >
              {getThemeIcon()}
              <span className="ml-3 capitalize">{theme} theme</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="lg:pl-64">
        {/* Top bar */}
        <div className="sticky top-0 z-30 flex h-16 items-center justify-between bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b border-border px-4 lg:px-6">
          <button
            onClick={() => setSidebarOpen(true)}
            className="lg:hidden p-2 rounded-md hover:bg-accent"
          >
            <Menu className="h-5 w-5" />
          </button>

          <div className="flex items-center space-x-4">
            {/* Breadcrumb or page title could go here */}
            <h1 className="text-lg font-semibold text-foreground">
              {navigation.find(item => isCurrentPath(item.href))?.name || 'Agentic AI'}
            </h1>
          </div>

          <div className="flex items-center space-x-4">
            {/* Connection indicator */}
            <div className="hidden sm:flex items-center space-x-2">
              <div
                className={`h-2 w-2 rounded-full ${
                  isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
                }`}
              />
              <span className="text-sm text-muted-foreground">
                {isConnected ? 'Live' : 'Offline'}
              </span>
            </div>

            {/* Theme toggle for mobile */}
            <button
              onClick={toggleTheme}
              className="p-2 rounded-md hover:bg-accent lg:hidden"
            >
              {getThemeIcon()}
            </button>
          </div>
        </div>

        {/* Page content */}
        <main className="flex-1 p-4 lg:p-6">
          <div className="mx-auto max-w-7xl">
            {children}
          </div>
        </main>
      </div>

      {/* Floating Debug Button */}
      {import.meta.env.DEV && (
        <button
          onClick={() => setDebugPanelOpen(true)}
          className="fixed bottom-4 left-4 bg-orange-600 hover:bg-orange-700 text-white p-3 rounded-full shadow-lg z-40 transition-colors"
          title="Open Debug Panel"
        >
          <Bug className="w-5 h-5" />
          {errors.length > 0 && (
            <span className="absolute -top-2 -right-2 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
              {errors.length > 9 ? '9+' : errors.length}
            </span>
          )}
        </button>
      )}

      {/* Debug Panel */}
      <DebugPanel
        isOpen={debugPanelOpen}
        onClose={() => setDebugPanelOpen(false)}
      />
    </div>
  )
}

export default Layout
