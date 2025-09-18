import React, { useState } from 'react'
import { useQuery } from 'react-query'
import { Plus, Play, GitBranch, Search, Filter, Palette } from 'lucide-react'
import { workflowApi } from '../services/api'
import WorkflowTemplateCard from '../components/Workflow/WorkflowTemplateCard'
import CreateWorkflowModal from '../components/Workflow/CreateWorkflowModal'
import WorkflowCanvas from '../components/Workflow/WorkflowCanvas'
import LangGraphDesigner from '../components/Workflow/LangGraphDesigner'

const WorkflowDesigner: React.FC = () => {
  const [view, setView] = useState<'templates' | 'designer' | 'history'>('templates')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showLangGraphDesigner, setShowLangGraphDesigner] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('all')

  // Fetch workflow templates
  const { data: templatesData, isLoading: templatesLoading } = useQuery(
    'workflow-templates',
    workflowApi.getTemplates
  )

  // Fetch workflow history
  const { data: historyData, isLoading: historyLoading } = useQuery(
    'workflow-history',
    () => workflowApi.getHistory({ limit: 20 })
  )

  const templates = templatesData?.templates || []
  const workflows = historyData?.workflows || []

  // Filter templates
  const filteredTemplates = templates.filter(template => {
    const matchesSearch = template.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         template.description.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesCategory = selectedCategory === 'all' || template.category === selectedCategory
    return matchesSearch && matchesCategory
  })

  // Get unique categories
  const categories = ['all', ...new Set(templates.map(t => t.category))]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Workflow Designer</h1>
          <p className="text-muted-foreground mt-1">
            Create and manage multi-agent workflows for complex tasks
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setView('designer')}
            className="btn-secondary inline-flex items-center"
          >
            <GitBranch className="h-4 w-4 mr-2" />
            Visual Designer
          </button>
          <button
            onClick={() => setShowLangGraphDesigner(true)}
            className="btn-secondary inline-flex items-center"
          >
            <Palette className="h-4 w-4 mr-2" />
            LangGraph Designer
          </button>
          <button
            onClick={() => setShowCreateModal(true)}
            className="btn-primary inline-flex items-center"
          >
            <Plus className="h-4 w-4 mr-2" />
            Create Workflow
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
          Templates
        </button>
        <button
          onClick={() => setView('designer')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            view === 'designer'
              ? 'bg-background text-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground'
          }`}
        >
          Designer
        </button>
        <button
          onClick={() => setView('history')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            view === 'history'
              ? 'bg-background text-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground'
          }`}
        >
          History ({workflows.length})
        </button>
      </div>

      {/* Content */}
      {view === 'templates' && (
        <div className="space-y-6">
          {/* Search and Filters */}
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search workflow templates..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="input pl-10"
              />
            </div>
            
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
          </div>

          {/* Templates Grid */}
          {templatesLoading ? (
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
                <WorkflowTemplateCard
                  key={template.id}
                  template={template}
                  onUseTemplate={(template) => {
                    // Handle template usage
                    console.log('Using template:', template)
                  }}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <GitBranch className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium text-foreground mb-2">No templates found</h3>
              <p className="text-muted-foreground">
                {searchTerm || selectedCategory !== 'all'
                  ? 'Try adjusting your search or filter criteria'
                  : 'No workflow templates available'}
              </p>
            </div>
          )}
        </div>
      )}

      {view === 'designer' && (
        <div className="space-y-6">
          <div className="card">
            <div className="card-header">
              <h2 className="text-xl font-semibold text-foreground">Visual Workflow Designer</h2>
              <p className="text-muted-foreground">
                Drag and drop agents to create complex workflows
              </p>
            </div>
            <div className="card-content">
              <WorkflowCanvas />
            </div>
          </div>
        </div>
      )}

      {view === 'history' && (
        <div className="space-y-6">
          <div className="card">
            <div className="card-header">
              <h2 className="text-xl font-semibold text-foreground">Workflow History</h2>
              <p className="text-muted-foreground">
                View and manage your workflow executions
              </p>
            </div>
            <div className="card-content">
              {historyLoading ? (
                <div className="space-y-4">
                  {[...Array(5)].map((_, i) => (
                    <div key={i} className="p-4 border border-border rounded-lg animate-pulse">
                      <div className="h-4 bg-muted rounded w-1/2 mb-2"></div>
                      <div className="h-3 bg-muted rounded w-3/4"></div>
                    </div>
                  ))}
                </div>
              ) : workflows.length > 0 ? (
                <div className="space-y-4">
                  {workflows.map((workflow) => (
                    <div key={workflow.id} className="p-4 border border-border rounded-lg hover:border-primary/50 transition-colors">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-medium text-foreground">{workflow.task}</h3>
                        <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                          workflow.status === 'completed' 
                            ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
                            : workflow.status === 'running'
                            ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400'
                            : 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
                        }`}>
                          {workflow.status}
                        </div>
                      </div>
                      <div className="flex items-center justify-between text-sm text-muted-foreground">
                        <span>Agents: {workflow.agents_used?.join(', ')}</span>
                        <span>
                          {workflow.execution_time 
                            ? `${workflow.execution_time}s`
                            : 'In progress'
                          }
                        </span>
                      </div>
                      {workflow.result_summary && (
                        <p className="text-sm text-muted-foreground mt-2">
                          {workflow.result_summary}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <GitBranch className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-foreground mb-2">No workflows executed yet</h3>
                  <p className="text-muted-foreground mb-4">
                    Create and execute your first workflow to see it here
                  </p>
                  <button
                    onClick={() => setView('templates')}
                    className="btn-primary inline-flex items-center"
                  >
                    <Play className="h-4 w-4 mr-2" />
                    Browse Templates
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Create Workflow Modal */}
      {showCreateModal && (
        <CreateWorkflowModal
          isOpen={showCreateModal}
          onClose={() => setShowCreateModal(false)}
          templates={templates}
        />
      )}

      {/* LangGraph Designer */}
      <LangGraphDesigner
        isOpen={showLangGraphDesigner}
        onClose={() => setShowLangGraphDesigner(false)}
      />
    </div>
  )
}

export default WorkflowDesigner
