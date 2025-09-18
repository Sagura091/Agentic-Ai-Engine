import React, { useState } from 'react'
import { useQuery } from 'react-query'
import {
  Settings as SettingsIcon,
  Save,
  Download,
  Upload,
  RefreshCw,
  Monitor,
  Sun,
  Moon,
  Bell,
  Shield,
  Database,
  Cpu,
  Zap,
  Activity
} from 'lucide-react'
import { useTheme } from '../contexts/ThemeContext'
import { settingsApi, modelsApi } from '../services/api'
import LLMProviderConfig from '../components/LLMProviderConfig'
import SystemStatus from '../components/SystemStatus'
import EmbeddingSettings from '../components/Settings/EmbeddingSettings'
import toast from 'react-hot-toast'

const Settings: React.FC = () => {
  const { theme, setTheme } = useTheme()
  const [activeTab, setActiveTab] = useState('general')
  const [isLoading, setIsLoading] = useState(false)

  // Fetch current settings
  const { data: settingsData } = useQuery('settings', settingsApi.getSettings)
  const { data: modelsData } = useQuery('models', modelsApi.getModels)

  // Settings state
  const [settings, setSettings] = useState({
    // General settings
    defaultModel: 'llama3.2:latest',
    maxConcurrentAgents: 10,
    agentTimeout: 300,
    enableMetrics: true,
    enableLogging: true,
    
    // UI settings
    theme: theme,
    enableAnimations: true,
    compactMode: false,
    showNotifications: true,
    
    // Performance settings
    maxTokens: 2048,
    temperature: 0.7,
    requestTimeout: 60,
    retryAttempts: 3,
    
    // Security settings
    enableCors: true,
    allowedOrigins: 'http://localhost:3000,http://localhost:5173',
    enableRateLimit: true,
    maxRequestsPerMinute: 100,
  })

  const tabs = [
    { id: 'general', name: 'General', icon: SettingsIcon },
    { id: 'embedding', name: 'Embedding & RAG', icon: Database },
    { id: 'providers', name: 'LLM Providers', icon: Zap },
    { id: 'system', name: 'System Status', icon: Activity },
    { id: 'appearance', name: 'Appearance', icon: Monitor },
    { id: 'performance', name: 'Performance', icon: Cpu },
    { id: 'security', name: 'Security', icon: Shield },
    { id: 'data', name: 'Data & Storage', icon: Database },
  ]

  const handleSaveSettings = async () => {
    try {
      setIsLoading(true)
      await settingsApi.updateSettings(settings)
      toast.success('Settings saved successfully!')
    } catch (error: any) {
      toast.error(error.message || 'Failed to save settings')
    } finally {
      setIsLoading(false)
    }
  }

  const handleExportConfig = async () => {
    try {
      const config = await settingsApi.exportConfig()
      const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'agentic-ai-config.json'
      a.click()
      URL.revokeObjectURL(url)
      toast.success('Configuration exported successfully!')
    } catch (error: any) {
      toast.error(error.message || 'Failed to export configuration')
    }
  }

  const handleImportConfig = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = async (e) => {
      try {
        const config = JSON.parse(e.target?.result as string)
        await settingsApi.importConfig(config)
        toast.success('Configuration imported successfully!')
        window.location.reload()
      } catch (error: any) {
        toast.error(error.message || 'Failed to import configuration')
      }
    }
    reader.readAsText(file)
  }

  const updateSetting = (key: string, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }))
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Settings</h1>
          <p className="text-muted-foreground mt-1">
            Configure your Agentic AI system preferences and behavior
          </p>
        </div>
        
        <div className="flex items-center space-x-3">
          <button
            onClick={handleExportConfig}
            className="btn-secondary inline-flex items-center"
          >
            <Download className="h-4 w-4 mr-2" />
            Export Config
          </button>
          
          <label className="btn-secondary inline-flex items-center cursor-pointer">
            <Upload className="h-4 w-4 mr-2" />
            Import Config
            <input
              type="file"
              accept=".json"
              onChange={handleImportConfig}
              className="hidden"
            />
          </label>
          
          <button
            onClick={handleSaveSettings}
            disabled={isLoading}
            className="btn-primary inline-flex items-center"
          >
            {isLoading ? (
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            Save Changes
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar */}
        <div className="lg:col-span-1">
          <nav className="space-y-1">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center space-x-3 px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                    activeTab === tab.id
                      ? 'bg-primary text-primary-foreground'
                      : 'text-muted-foreground hover:text-foreground hover:bg-accent'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{tab.name}</span>
                </button>
              )
            })}
          </nav>
        </div>

        {/* Content */}
        <div className="lg:col-span-3">
          <div className="card">
            <div className="card-content p-6">
              {activeTab === 'general' && (
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-medium text-foreground mb-4">General Settings</h3>
                    
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-foreground mb-2">
                          Default Model
                        </label>
                        <select
                          value={settings.defaultModel}
                          onChange={(e) => updateSetting('defaultModel', e.target.value)}
                          className="input"
                        >
                          {modelsData?.available_models?.map((model: string) => (
                            <option key={model} value={model}>{model}</option>
                          ))}
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-foreground mb-2">
                          Max Concurrent Agents: {settings.maxConcurrentAgents}
                        </label>
                        <input
                          type="range"
                          min="1"
                          max="50"
                          value={settings.maxConcurrentAgents}
                          onChange={(e) => updateSetting('maxConcurrentAgents', parseInt(e.target.value))}
                          className="w-full"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-foreground mb-2">
                          Agent Timeout (seconds): {settings.agentTimeout}
                        </label>
                        <input
                          type="range"
                          min="30"
                          max="1800"
                          value={settings.agentTimeout}
                          onChange={(e) => updateSetting('agentTimeout', parseInt(e.target.value))}
                          className="w-full"
                        />
                      </div>

                      <div className="flex items-center justify-between">
                        <div>
                          <label className="text-sm font-medium text-foreground">Enable Metrics</label>
                          <p className="text-xs text-muted-foreground">Collect performance metrics</p>
                        </div>
                        <input
                          type="checkbox"
                          checked={settings.enableMetrics}
                          onChange={(e) => updateSetting('enableMetrics', e.target.checked)}
                          className="rounded"
                        />
                      </div>

                      <div className="flex items-center justify-between">
                        <div>
                          <label className="text-sm font-medium text-foreground">Enable Logging</label>
                          <p className="text-xs text-muted-foreground">Log system activities</p>
                        </div>
                        <input
                          type="checkbox"
                          checked={settings.enableLogging}
                          onChange={(e) => updateSetting('enableLogging', e.target.checked)}
                          className="rounded"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'embedding' && (
                <div className="space-y-6">
                  <EmbeddingSettings />
                </div>
              )}

              {activeTab === 'providers' && (
                <div className="space-y-6">
                  <LLMProviderConfig />
                </div>
              )}

              {activeTab === 'system' && (
                <div className="space-y-6">
                  <SystemStatus showDetails={true} />
                </div>
              )}

              {activeTab === 'appearance' && (
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-medium text-foreground mb-4">Appearance Settings</h3>
                    
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-foreground mb-2">
                          Theme
                        </label>
                        <div className="flex items-center space-x-3">
                          <button
                            onClick={() => setTheme('light')}
                            className={`flex items-center space-x-2 p-3 rounded-lg border transition-colors ${
                              theme === 'light' 
                                ? 'border-primary bg-primary/10' 
                                : 'border-border hover:bg-accent'
                            }`}
                          >
                            <Sun className="h-4 w-4" />
                            <span>Light</span>
                          </button>
                          <button
                            onClick={() => setTheme('dark')}
                            className={`flex items-center space-x-2 p-3 rounded-lg border transition-colors ${
                              theme === 'dark' 
                                ? 'border-primary bg-primary/10' 
                                : 'border-border hover:bg-accent'
                            }`}
                          >
                            <Moon className="h-4 w-4" />
                            <span>Dark</span>
                          </button>
                          <button
                            onClick={() => setTheme('system')}
                            className={`flex items-center space-x-2 p-3 rounded-lg border transition-colors ${
                              theme === 'system' 
                                ? 'border-primary bg-primary/10' 
                                : 'border-border hover:bg-accent'
                            }`}
                          >
                            <Monitor className="h-4 w-4" />
                            <span>System</span>
                          </button>
                        </div>
                      </div>

                      <div className="flex items-center justify-between">
                        <div>
                          <label className="text-sm font-medium text-foreground">Enable Animations</label>
                          <p className="text-xs text-muted-foreground">Smooth transitions and effects</p>
                        </div>
                        <input
                          type="checkbox"
                          checked={settings.enableAnimations}
                          onChange={(e) => updateSetting('enableAnimations', e.target.checked)}
                          className="rounded"
                        />
                      </div>

                      <div className="flex items-center justify-between">
                        <div>
                          <label className="text-sm font-medium text-foreground">Compact Mode</label>
                          <p className="text-xs text-muted-foreground">Reduce spacing and padding</p>
                        </div>
                        <input
                          type="checkbox"
                          checked={settings.compactMode}
                          onChange={(e) => updateSetting('compactMode', e.target.checked)}
                          className="rounded"
                        />
                      </div>

                      <div className="flex items-center justify-between">
                        <div>
                          <label className="text-sm font-medium text-foreground">Show Notifications</label>
                          <p className="text-xs text-muted-foreground">Display toast notifications</p>
                        </div>
                        <input
                          type="checkbox"
                          checked={settings.showNotifications}
                          onChange={(e) => updateSetting('showNotifications', e.target.checked)}
                          className="rounded"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'performance' && (
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-medium text-foreground mb-4">Performance Settings</h3>
                    
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-foreground mb-2">
                          Max Tokens: {settings.maxTokens}
                        </label>
                        <input
                          type="range"
                          min="256"
                          max="8192"
                          value={settings.maxTokens}
                          onChange={(e) => updateSetting('maxTokens', parseInt(e.target.value))}
                          className="w-full"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-foreground mb-2">
                          Temperature: {settings.temperature}
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="2"
                          step="0.1"
                          value={settings.temperature}
                          onChange={(e) => updateSetting('temperature', parseFloat(e.target.value))}
                          className="w-full"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-foreground mb-2">
                          Request Timeout (seconds): {settings.requestTimeout}
                        </label>
                        <input
                          type="range"
                          min="10"
                          max="300"
                          value={settings.requestTimeout}
                          onChange={(e) => updateSetting('requestTimeout', parseInt(e.target.value))}
                          className="w-full"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-foreground mb-2">
                          Retry Attempts: {settings.retryAttempts}
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="10"
                          value={settings.retryAttempts}
                          onChange={(e) => updateSetting('retryAttempts', parseInt(e.target.value))}
                          className="w-full"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'security' && (
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-medium text-foreground mb-4">Security Settings</h3>
                    
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <label className="text-sm font-medium text-foreground">Enable CORS</label>
                          <p className="text-xs text-muted-foreground">Allow cross-origin requests</p>
                        </div>
                        <input
                          type="checkbox"
                          checked={settings.enableCors}
                          onChange={(e) => updateSetting('enableCors', e.target.checked)}
                          className="rounded"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-foreground mb-2">
                          Allowed Origins
                        </label>
                        <input
                          type="text"
                          value={settings.allowedOrigins}
                          onChange={(e) => updateSetting('allowedOrigins', e.target.value)}
                          className="input"
                          placeholder="http://localhost:3000,http://localhost:5173"
                        />
                      </div>

                      <div className="flex items-center justify-between">
                        <div>
                          <label className="text-sm font-medium text-foreground">Enable Rate Limiting</label>
                          <p className="text-xs text-muted-foreground">Limit requests per minute</p>
                        </div>
                        <input
                          type="checkbox"
                          checked={settings.enableRateLimit}
                          onChange={(e) => updateSetting('enableRateLimit', e.target.checked)}
                          className="rounded"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-foreground mb-2">
                          Max Requests Per Minute: {settings.maxRequestsPerMinute}
                        </label>
                        <input
                          type="range"
                          min="10"
                          max="1000"
                          value={settings.maxRequestsPerMinute}
                          onChange={(e) => updateSetting('maxRequestsPerMinute', parseInt(e.target.value))}
                          className="w-full"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'data' && (
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-medium text-foreground mb-4">Data & Storage Settings</h3>
                    
                    <div className="space-y-4">
                      <div className="p-4 border border-border rounded-lg">
                        <h4 className="font-medium text-foreground mb-2">Storage Usage</h4>
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span>Agent Configurations</span>
                            <span>2.3 MB</span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span>Conversation History</span>
                            <span>15.7 MB</span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span>Workflow Data</span>
                            <span>8.1 MB</span>
                          </div>
                          <div className="flex justify-between text-sm font-medium">
                            <span>Total</span>
                            <span>26.1 MB</span>
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center justify-between p-4 border border-border rounded-lg">
                        <div>
                          <h4 className="font-medium text-foreground">Clear Conversation History</h4>
                          <p className="text-xs text-muted-foreground">Remove all chat history</p>
                        </div>
                        <button className="btn-secondary text-destructive hover:bg-destructive/10">
                          Clear History
                        </button>
                      </div>

                      <div className="flex items-center justify-between p-4 border border-border rounded-lg">
                        <div>
                          <h4 className="font-medium text-foreground">Reset All Settings</h4>
                          <p className="text-xs text-muted-foreground">Restore default configuration</p>
                        </div>
                        <button className="btn-secondary text-destructive hover:bg-destructive/10">
                          Reset Settings
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Settings
