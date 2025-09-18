/**
 * Real-time Agent Monitoring System
 * 
 * Provides comprehensive monitoring of agent activities, performance metrics,
 * and system health with real-time updates via WebSocket connections.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { useSocket } from '../../contexts/SocketContext';
import { useError } from '../../contexts/ErrorContext';
import { buildApiUrl } from '../../utils/backendAlignment';

interface AgentMetrics {
  agentId: string;
  name: string;
  status: 'active' | 'idle' | 'error' | 'offline';
  tasksCompleted: number;
  tasksInProgress: number;
  averageResponseTime: number;
  successRate: number;
  lastActivity: string;
  memoryUsage: number;
  cpuUsage: number;
}

interface SystemMetrics {
  totalAgents: number;
  activeAgents: number;
  totalTasks: number;
  completedTasks: number;
  failedTasks: number;
  averageSystemLoad: number;
  memoryUsage: {
    used: number;
    total: number;
    percentage: number;
  };
  diskSpace: {
    used: number;
    total: number;
    percentage: number;
  };
}

interface MonitoringData {
  agents: AgentMetrics[];
  system: SystemMetrics;
  timestamp: string;
}

export const RealTimeMonitor: React.FC = () => {
  const [monitoringData, setMonitoringData] = useState<MonitoringData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(5000); // 5 seconds
  const { socket, isConnected: socketConnected } = useSocket();
  const { addError } = useError();

  // Handle real-time monitoring data updates
  const handleMonitoringUpdate = useCallback((data: MonitoringData) => {
    setMonitoringData(data);
  }, []);

  // Handle connection status changes
  const handleConnectionChange = useCallback((connected: boolean) => {
    setIsConnected(connected);
    if (!connected) {
      addError({
        type: 'network',
        message: 'Lost connection to monitoring service',
        severity: 'medium',
        location: 'RealTimeMonitor'
      });
    }
  }, [addError]);

  // Setup WebSocket listeners
  useEffect(() => {
    if (!socket || !socketConnected) return;

    // Join monitoring room
    socket.emit('join_monitoring');

    // Listen for monitoring updates
    socket.on('monitoring_update', handleMonitoringUpdate);
    socket.on('agent_status_change', (agentUpdate: Partial<AgentMetrics>) => {
      setMonitoringData(prev => {
        if (!prev) return prev;
        
        const updatedAgents = prev.agents.map(agent => 
          agent.agentId === agentUpdate.agentId 
            ? { ...agent, ...agentUpdate }
            : agent
        );
        
        return { ...prev, agents: updatedAgents };
      });
    });

    // Handle connection events
    socket.on('connect', () => handleConnectionChange(true));
    socket.on('disconnect', () => handleConnectionChange(false));

    // Initial connection status
    setIsConnected(socket.connected);

    return () => {
      socket.off('monitoring_update', handleMonitoringUpdate);
      socket.off('agent_status_change');
      socket.off('connect');
      socket.off('disconnect');
      socket.emit('leave_monitoring');
    };
  }, [socket, socketConnected, handleMonitoringUpdate, handleConnectionChange]);

  // Fetch initial monitoring data
  const fetchMonitoringData = useCallback(async () => {
    try {
      const response = await fetch(buildApiUrl('/monitoring/dashboard'), {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setMonitoringData(data);
    } catch (error) {
      addError({
        type: 'network',
        message: `Failed to fetch monitoring data: ${error instanceof Error ? error.message : 'Unknown error'}`,
        severity: 'high',
        location: 'RealTimeMonitor.fetchMonitoringData'
      });
    }
  }, [addError]);

  // Initial data fetch and periodic refresh fallback
  useEffect(() => {
    fetchMonitoringData();

    // Fallback polling if WebSocket is not available
    if (!socketConnected) {
      const interval = setInterval(fetchMonitoringData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [fetchMonitoringData, socketConnected, refreshInterval]);

  // Render connection status indicator
  const renderConnectionStatus = () => (
    <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
      <div className={`status-indicator ${isConnected ? 'green' : 'red'}`} />
      <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
      {!isConnected && (
        <button onClick={fetchMonitoringData} className="retry-button">
          Retry
        </button>
      )}
    </div>
  );

  // Render agent metrics card
  const renderAgentCard = (agent: AgentMetrics) => (
    <div key={agent.agentId} className={`agent-card ${agent.status}`}>
      <div className="agent-header">
        <h3>{agent.name}</h3>
        <span className={`status-badge ${agent.status}`}>{agent.status}</span>
      </div>
      
      <div className="agent-metrics">
        <div className="metric">
          <label>Tasks Completed:</label>
          <span>{agent.tasksCompleted}</span>
        </div>
        <div className="metric">
          <label>In Progress:</label>
          <span>{agent.tasksInProgress}</span>
        </div>
        <div className="metric">
          <label>Success Rate:</label>
          <span>{(agent.successRate * 100).toFixed(1)}%</span>
        </div>
        <div className="metric">
          <label>Avg Response:</label>
          <span>{agent.averageResponseTime.toFixed(0)}ms</span>
        </div>
        <div className="metric">
          <label>Memory:</label>
          <span>{agent.memoryUsage.toFixed(1)}MB</span>
        </div>
        <div className="metric">
          <label>CPU:</label>
          <span>{agent.cpuUsage.toFixed(1)}%</span>
        </div>
      </div>
      
      <div className="last-activity">
        Last Activity: {new Date(agent.lastActivity).toLocaleTimeString()}
      </div>
    </div>
  );

  // Render system metrics
  const renderSystemMetrics = (system: SystemMetrics) => (
    <div className="system-metrics">
      <h2>System Overview</h2>
      
      <div className="metrics-grid">
        <div className="metric-card">
          <h3>Agents</h3>
          <div className="metric-value">{system.activeAgents}/{system.totalAgents}</div>
          <div className="metric-label">Active/Total</div>
        </div>
        
        <div className="metric-card">
          <h3>Tasks</h3>
          <div className="metric-value">{system.completedTasks}/{system.totalTasks}</div>
          <div className="metric-label">Completed/Total</div>
        </div>
        
        <div className="metric-card">
          <h3>Success Rate</h3>
          <div className="metric-value">
            {((system.completedTasks / Math.max(system.totalTasks, 1)) * 100).toFixed(1)}%
          </div>
          <div className="metric-label">Overall</div>
        </div>
        
        <div className="metric-card">
          <h3>System Load</h3>
          <div className="metric-value">{system.averageSystemLoad.toFixed(2)}</div>
          <div className="metric-label">Average</div>
        </div>
        
        <div className="metric-card">
          <h3>Memory</h3>
          <div className="metric-value">{system.memoryUsage.percentage.toFixed(1)}%</div>
          <div className="metric-label">
            {(system.memoryUsage.used / 1024).toFixed(1)}GB / {(system.memoryUsage.total / 1024).toFixed(1)}GB
          </div>
        </div>
        
        <div className="metric-card">
          <h3>Disk Space</h3>
          <div className="metric-value">{system.diskSpace.percentage.toFixed(1)}%</div>
          <div className="metric-label">
            {(system.diskSpace.used / 1024).toFixed(1)}GB / {(system.diskSpace.total / 1024).toFixed(1)}GB
          </div>
        </div>
      </div>
    </div>
  );

  if (!monitoringData) {
    return (
      <div className="monitoring-container loading">
        <div className="loading-spinner" />
        <p>Loading monitoring data...</p>
        {renderConnectionStatus()}
      </div>
    );
  }

  return (
    <div className="monitoring-container">
      <div className="monitoring-header">
        <h1>Real-time Agent Monitoring</h1>
        {renderConnectionStatus()}
        
        <div className="controls">
          <label>
            Refresh Interval:
            <select 
              value={refreshInterval} 
              onChange={(e) => setRefreshInterval(Number(e.target.value))}
            >
              <option value={1000}>1 second</option>
              <option value={5000}>5 seconds</option>
              <option value={10000}>10 seconds</option>
              <option value={30000}>30 seconds</option>
            </select>
          </label>
        </div>
      </div>

      {renderSystemMetrics(monitoringData.system)}

      <div className="agents-section">
        <h2>Agent Status ({monitoringData.agents.length})</h2>
        <div className="agents-grid">
          {monitoringData.agents.map(renderAgentCard)}
        </div>
      </div>

      <div className="monitoring-footer">
        <p>Last Updated: {new Date(monitoringData.timestamp).toLocaleString()}</p>
      </div>
    </div>
  );
};

export default RealTimeMonitor;
