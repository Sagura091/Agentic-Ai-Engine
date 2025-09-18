import React, { useState } from 'react'
import { useQuery } from 'react-query'
import { Plus, Search, Filter, Bot, Sparkles, Wrench, Palette } from 'lucide-react'
import { useAgent } from '../contexts/AgentContext'
import { modelsApi } from '../services/api'
import AgentTemplateCard from '../components/Agent/AgentTemplateCard'
import CreateAgentModal from '../components/Agent/CreateAgentModal'
import AgentCard from '../components/Agent/AgentCard'
import VisualAgentBuilder from '../components/Agent/VisualAgentBuilder'
import CustomToolBuilder from '../components/Agent/CustomToolBuilder'

const AgentBuilder: React.FC = () => {
  const { agents, templates, isLoading } = useAgent()
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showVisualBuilder, setShowVisualBuilder] = useState(false)
  const [showToolBuilder, setShowToolBuilder] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [view, setView] = useState<'templates' | 'agents'>('templates')
  const [editingAgent, setEditingAgent] = useState(null)

  // Fetch available models
  const { data: modelsData } = useQuery('models', modelsApi.getModels)

  // Filter templates
  const filteredTemplates = templates.filter(template => {
    const matchesSearch = template.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         template.description.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesCategory = selectedCategory === 'all' || template.category === selectedCategory
    return matchesSearch && matchesCategory
  })

  // Filter agents
  const filteredAgents = agents.filter(agent => {
    const matchesSearch = agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         agent.description.toLowerCase().includes(searchTerm.toLowerCase())
    return matchesSearch
  })

  // Get unique categories
  const categories = ['all', ...new Set(templates.map(t => t.category))]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Agent Builder</h1>
          <p className="text-muted-foreground mt-1">
            Create and manage your AI agents with custom capabilities
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowToolBuilder(true)}
            className="btn-secondary inline-flex items-center"
          >
            <Wrench className="h-4 w-4 mr-2" />
            Custom Tools
          </button>
          <button
            onClick={() => setShowVisualBuilder(true)}
            className="btn-secondary inline-flex items-center"
          >
            <Palette className="h-4 w-4 mr-2" />
            Visual Builder
          </button>
          <button
            onClick={() => setShowCreateModal(true)}
            className="btn-primary inline-flex items-center"
          >
            <Plus className="h-4 w-4 mr-2" />
            Create Agent
          </button>
        </div>
      </div>

      {/* View Toggle */}
      <div className="flex items-center space-x-1 bg-muted p-1 rounded-lg w-fit">
        <button
          onClick={() => setView('templates')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            view === 'templates'
              ? 'bg-background text-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground'
          }`}
        >
          <Sparkles className="h-4 w-4 mr-2 inline" />
          Templates
        </button>
        <button
          onClick={() => setView('agents')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            view === 'agents'
              ? 'bg-background text-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground'
          }`}
        >
          <Bot className="h-4 w-4 mr-2 inline" />
          My Agents ({agents.length})
        </button>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            placeholder={`Search ${view}...`}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="input pl-10"
          />
        </div>
        
        {view === 'templates' && (
          <div className="relative">
            <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="input pl-10 pr-8 appearance-none bg-background"
            >
              {categories.map(category => (
                <option key={category} value={category}>
                  {category === 'all' ? 'All Categories' : category}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Content */}
      {view === 'templates' ? (
        <div>
          {/* Templates Grid */}
          {isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[...Array(6)].map((_, i) => (
                <div key={i} className="card p-6 animate-pulse">
                  <div className="h-4 bg-muted rounded w-3/4 mb-2"></div>
                  <div className="h-3 bg-muted rounded w-full mb-4"></div>
                  <div className="h-3 bg-muted rounded w-1/2"></div>
                </div>
              ))}
            </div>
          ) : filteredTemplates.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredTemplates.map((template) => (
                <AgentTemplateCard
                  key={template.id}
                  template={template}
                  onUseTemplate={() => setShowCreateModal(true)}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <Sparkles className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium text-foreground mb-2">No templates found</h3>
              <p className="text-muted-foreground">
                {searchTerm || selectedCategory !== 'all'
                  ? 'Try adjusting your search or filter criteria'
                  : 'No agent templates available'}
              </p>
            </div>
          )}
        </div>
      ) : (
        <div>
          {/* Agents Grid */}
          {isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[...Array(3)].map((_, i) => (
                <div key={i} className="card p-6 animate-pulse">
                  <div className="h-4 bg-muted rounded w-3/4 mb-2"></div>
                  <div className="h-3 bg-muted rounded w-full mb-4"></div>
                  <div className="h-3 bg-muted rounded w-1/2"></div>
                </div>
              ))}
            </div>
          ) : filteredAgents.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredAgents.map((agent) => (
                <AgentCard
                  key={agent.id}
                  agent={agent}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <Bot className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium text-foreground mb-2">
                {searchTerm ? 'No agents found' : 'No agents created yet'}
              </h3>
              <p className="text-muted-foreground mb-4">
                {searchTerm 
                  ? 'Try adjusting your search criteria'
                  : 'Create your first agent to get started with AI automation'}
              </p>
              <button
                onClick={() => setShowCreateModal(true)}
                className="btn-primary inline-flex items-center"
              >
                <Plus className="h-4 w-4 mr-2" />
                Create Your First Agent
              </button>
            </div>
          )}
        </div>
      )}

      {/* Create Agent Modal */}
      {showCreateModal && (
        <CreateAgentModal
          isOpen={showCreateModal}
          onClose={() => setShowCreateModal(false)}
          templates={templates}
        />
      )}

      {/* Visual Agent Builder */}
      <VisualAgentBuilder
        isOpen={showVisualBuilder}
        onClose={() => setShowVisualBuilder(false)}
        editingAgent={editingAgent}
      />

      {/* Custom Tool Builder */}
      <CustomToolBuilder
        isOpen={showToolBuilder}
        onClose={() => setShowToolBuilder(false)}
        onSave={(tool) => {
          console.log('Custom tool created:', tool)
          // In a real implementation, this would save to backend
        }}
      />
    </div>
  )
}

export default AgentBuilder
