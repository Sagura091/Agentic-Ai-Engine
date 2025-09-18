
import React, { useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from 'react-query'
import { Toaster } from 'react-hot-toast'
import { ThemeProvider } from './contexts/ThemeContext'
import { SocketProvider } from './contexts/SocketContext'
import { AgentProvider } from './contexts/AgentContext'
import { ErrorProvider } from './contexts/ErrorContext'

// Layout Components
import Layout from './components/Layout/Layout'
import LogViewer from './components/LogViewer'

// Page Components
import Dashboard from './pages/Dashboard'
import AgentBuilder from './pages/AgentBuilder'
import WorkflowDesigner from './pages/WorkflowDesigner'
import AgentChat from './pages/AgentChat'
import Monitoring from './pages/Monitoring'
import Settings from './pages/Settings'
import ConversationalAgents from './pages/ConversationalAgents'
import RAGManagement from './pages/RAGManagement'
import KnowledgeBaseManagement from './pages/KnowledgeBaseManagement'
import KnowledgeBaseDetail from './pages/KnowledgeBaseDetail'

// Revolutionary RAG 4.0 Components
import AdvancedAnalyticsDashboard from './components/RAG/AdvancedAnalyticsDashboard'
import CollaborativeWorkspace from './components/RAG/CollaborativeWorkspace'
import PerformanceMonitoringDashboard from './components/RAG/PerformanceMonitoringDashboard'
import KnowledgeGraphViewer from './components/RAG/KnowledgeGraphViewer'
import MultiModalUploader from './components/RAG/MultiModalUploader'
import RevolutionarySearchInterface from './components/RAG/RevolutionarySearchInterface'
import ContextualSearchInterface from './components/RAG/ContextualSearchInterface'

// New Conversational Agent Components
import ConversationalAgentCreator from './components/ConversationalAgentCreator'
import AutonomousTaskExecutor from './components/AutonomousTaskExecutor'
import AgentTaskExecutor from './components/Agent/AgentTaskExecutor'

// Error Boundary Component
import ErrorBoundary from './components/ErrorBoundary'

// Styles
import './index.css'

// Create a client for React Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
  },
})

function App() {
  const [isLogViewerOpen, setIsLogViewerOpen] = useState(false)

  return (
    <QueryClientProvider client={queryClient}>
      <ErrorProvider>
        <ThemeProvider>
          <SocketProvider>
            <AgentProvider>
              <Router>
              <div className="min-h-screen bg-background text-foreground">
                <ErrorBoundary>
                  <Layout>
                    <Routes>
                      <Route path="/" element={<Dashboard />} />
                      <Route path="/dashboard" element={<Dashboard />} />
                      <Route path="/agents" element={<AgentBuilder />} />
                      <Route path="/conversational-agents" element={<ConversationalAgents />} />
                      <Route path="/conversational-creator" element={<ConversationalAgentCreator />} />
                      <Route path="/autonomous-executor" element={<AutonomousTaskExecutor />} />
                      <Route path="/agent-tasks" element={<AgentTaskExecutor />} />
                      <Route path="/rag" element={<RAGManagement />} />
                      <Route path="/knowledge-bases" element={<KnowledgeBaseManagement />} />
                      <Route path="/knowledge-bases/:kbId" element={<KnowledgeBaseDetail />} />

                      {/* Revolutionary RAG 4.0 Routes */}
                      <Route path="/rag/analytics" element={<AdvancedAnalyticsDashboard />} />
                      <Route path="/rag/collaboration" element={<CollaborativeWorkspace />} />
                      <Route path="/rag/performance" element={<PerformanceMonitoringDashboard />} />
                      <Route path="/rag/knowledge-graph" element={<KnowledgeGraphViewer />} />
                      <Route path="/rag/multimodal" element={<MultiModalUploader />} />
                      <Route path="/rag/search" element={<RevolutionarySearchInterface />} />
                      <Route path="/rag/contextual-search" element={<ContextualSearchInterface />} />

                      <Route path="/workflows" element={<WorkflowDesigner />} />
                      <Route path="/chat" element={<AgentChat />} />
                      <Route path="/chat/:agentId" element={<AgentChat />} />
                      <Route path="/monitoring" element={<Monitoring />} />
                      <Route path="/settings" element={<Settings />} />
                    </Routes>
                  </Layout>
                </ErrorBoundary>

                {/* Global Toast Notifications */}
                <Toaster
                  position="top-right"
                  toastOptions={{
                    duration: 4000,
                    className: 'bg-card text-card-foreground border border-border',
                    style: {
                      background: 'hsl(var(--card))',
                      color: 'hsl(var(--card-foreground))',
                      border: '1px solid hsl(var(--border))',
                    },
                  }}
                />

                {/* Floating Log Viewer Button */}
                <button
                  onClick={() => setIsLogViewerOpen(true)}
                  className="fixed bottom-4 right-4 bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-full shadow-lg z-40 transition-colors"
                  title="View Frontend Logs"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </button>

                {/* Log Viewer Modal */}
                <LogViewer
                  isOpen={isLogViewerOpen}
                  onClose={() => setIsLogViewerOpen(false)}
                />
              </div>
              </Router>
            </AgentProvider>
          </SocketProvider>
        </ThemeProvider>
      </ErrorProvider>
    </QueryClientProvider>
  )
}

export default App
