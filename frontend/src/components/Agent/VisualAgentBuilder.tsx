import React, { useState, useCallback } from 'react'
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd'
import { 
  Bot, 
  Plus, 
  Settings, 
  Code, 
  Wrench, 
  Trash2, 
  Move,
  Eye,
  Save,
  Play
} from 'lucide-react'
import { useAgent } from '../../contexts/AgentContext'
import toast from 'react-hot-toast'

interface AgentComponent {
  id: string
  type: 'capability' | 'tool' | 'model' | 'prompt'
  name: string
  description: string
  config: any
  position: { x: number; y: number }
}

interface VisualAgentBuilderProps {
  isOpen: boolean
  onClose: () => void
  editingAgent?: any
}

const VisualAgentBuilder: React.FC<VisualAgentBuilderProps> = ({
  isOpen,
  onClose,
  editingAgent
}) => {
  const { createAgent, updateAgent } = useAgent()
  const [agentName, setAgentName] = useState(editingAgent?.name || '')
  const [agentDescription, setAgentDescription] = useState(editingAgent?.description || '')
  const [components, setComponents] = useState<AgentComponent[]>([])
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null)
  const [showComponentLibrary, setShowComponentLibrary] = useState(false)

  const componentLibrary = [
    {
      type: 'capability',
      name: 'Reasoning',
      description: 'Logical reasoning and problem-solving',
      icon: '🧠',
      config: { type: 'reasoning', strength: 0.8 }
    },
    {
      type: 'capability',
      name: 'Research',
      description: 'Information gathering and analysis',
      icon: '🔍',
      config: { type: 'research', depth: 'comprehensive' }
    },
    {
      type: 'capability',
      name: 'Code Generation',
      description: 'Programming and code creation',
      icon: '💻',
      config: { type: 'coding', languages: ['python', 'javascript', 'typescript'] }
    },
    {
      type: 'tool',
      name: 'Web Search',
      description: 'Search the internet for information',
      icon: '🌐',
      config: { 
        name: 'web_search',
        parameters: { query: 'string', max_results: 'number' }
      }
    },
    {
      type: 'tool',
      name: 'Calculator',
      description: 'Perform mathematical calculations',
      icon: '🧮',
      config: {
        name: 'calculator',
        parameters: { expression: 'string' }
      }
    },
    {
      type: 'tool',
      name: 'File Reader',
      description: 'Read and analyze files',
      icon: '📄',
      config: {
        name: 'file_reader',
        parameters: { file_path: 'string', format: 'string' }
      }
    },
    {
      type: 'model',
      name: 'Llama 3.2',
      description: 'Latest Llama model for general tasks',
      icon: '🦙',
      config: { model: 'llama3.2:latest', temperature: 0.7 }
    },
    {
      type: 'model',
      name: 'Qwen 2.5',
      description: 'Qwen model for multilingual tasks',
      icon: '🌏',
      config: { model: 'qwen2.5:latest', temperature: 0.7 }
    }
  ]

  const addComponent = useCallback((libraryItem: any) => {
    const newComponent: AgentComponent = {
      id: `${libraryItem.type}_${Date.now()}`,
      type: libraryItem.type,
      name: libraryItem.name,
      description: libraryItem.description,
      config: { ...libraryItem.config },
      position: { 
        x: Math.random() * 400 + 50, 
        y: Math.random() * 300 + 50 
      }
    }
    setComponents(prev => [...prev, newComponent])
    setShowComponentLibrary(false)
  }, [])

  const removeComponent = useCallback((componentId: string) => {
    setComponents(prev => prev.filter(c => c.id !== componentId))
    if (selectedComponent === componentId) {
      setSelectedComponent(null)
    }
  }, [selectedComponent])

  const updateComponentConfig = useCallback((componentId: string, config: any) => {
    setComponents(prev => prev.map(c => 
      c.id === componentId ? { ...c, config: { ...c.config, ...config } } : c
    ))
  }, [])

  const handleDragEnd = (result: any) => {
    if (!result.destination) return

    const items = Array.from(components)
    const [reorderedItem] = items.splice(result.source.index, 1)
    items.splice(result.destination.index, 0, reorderedItem)

    setComponents(items)
  }

  const saveAgent = async () => {
    if (!agentName.trim()) {
      toast.error('Agent name is required')
      return
    }

    try {
      const agentConfig = {
        name: agentName,
        description: agentDescription,
        agent_type: 'custom',
        model: components.find(c => c.type === 'model')?.config.model || 'llama3.2:latest',
        capabilities: components.filter(c => c.type === 'capability').map(c => c.config.type),
        tools: components.filter(c => c.type === 'tool').map(c => c.config.name),
        system_prompt: generateSystemPrompt(),
        temperature: components.find(c => c.type === 'model')?.config.temperature || 0.7,
        max_tokens: 2048
      }

      if (editingAgent) {
        await updateAgent(editingAgent.id, agentConfig)
        toast.success('Agent updated successfully!')
      } else {
        await createAgent(agentConfig)
        toast.success('Agent created successfully!')
      }
      
      onClose()
    } catch (error: any) {
      toast.error(error.message || 'Failed to save agent')
    }
  }

  const generateSystemPrompt = () => {
    const capabilities = components.filter(c => c.type === 'capability')
    const tools = components.filter(c => c.type === 'tool')
    
    let prompt = `You are a specialized AI agent named "${agentName}".`
    
    if (agentDescription) {
      prompt += ` ${agentDescription}`
    }
    
    if (capabilities.length > 0) {
      prompt += ` You have the following capabilities: ${capabilities.map(c => c.name.toLowerCase()).join(', ')}.`
    }
    
    if (tools.length > 0) {
      prompt += ` You have access to these tools: ${tools.map(t => t.name).join(', ')}.`
    }
    
    prompt += ' Always be helpful, accurate, and follow best practices in your responses.'
    
    return prompt
  }

  const testAgent = async () => {
    if (!agentName.trim()) {
      toast.error('Please configure the agent first')
      return
    }
    
    toast.success('Agent test functionality would be implemented here')
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 overflow-hidden">
      <div className="flex h-full">
        {/* Component Library Sidebar */}
        <div className={`w-80 bg-card border-r border-border transition-transform duration-300 ${
          showComponentLibrary ? 'translate-x-0' : '-translate-x-full'
        }`}>
          <div className="p-4 border-b border-border">
            <h3 className="text-lg font-semibold text-foreground">Component Library</h3>
            <p className="text-sm text-muted-foreground">Drag components to build your agent</p>
          </div>
          
          <div className="p-4 space-y-3 overflow-y-auto h-full">
            {componentLibrary.map((item, index) => (
              <div
                key={index}
                className="p-3 border border-border rounded-lg hover:border-primary cursor-pointer transition-colors"
                onClick={() => addComponent(item)}
              >
                <div className="flex items-center space-x-3">
                  <span className="text-2xl">{item.icon}</span>
                  <div>
                    <h4 className="font-medium text-foreground">{item.name}</h4>
                    <p className="text-xs text-muted-foreground">{item.description}</p>
                    <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium mt-1 ${
                      item.type === 'capability' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400' :
                      item.type === 'tool' ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400' :
                      'bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-400'
                    }`}>
                      {item.type}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Main Canvas */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <div className="p-4 border-b border-border bg-card">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <button
                  onClick={() => setShowComponentLibrary(!showComponentLibrary)}
                  className="btn-secondary"
                >
                  <Plus className="h-4 w-4 mr-2" />
                  Components
                </button>
                
                <div className="flex items-center space-x-2">
                  <Bot className="h-5 w-5 text-primary" />
                  <input
                    type="text"
                    value={agentName}
                    onChange={(e) => setAgentName(e.target.value)}
                    placeholder="Agent Name"
                    className="input w-48"
                  />
                </div>
              </div>
              
              <div className="flex items-center space-x-2">
                <button onClick={testAgent} className="btn-secondary">
                  <Play className="h-4 w-4 mr-2" />
                  Test
                </button>
                <button onClick={saveAgent} className="btn-primary">
                  <Save className="h-4 w-4 mr-2" />
                  Save Agent
                </button>
                <button onClick={onClose} className="btn-ghost">
                  Close
                </button>
              </div>
            </div>
            
            <div className="mt-3">
              <input
                type="text"
                value={agentDescription}
                onChange={(e) => setAgentDescription(e.target.value)}
                placeholder="Agent description..."
                className="input w-full"
              />
            </div>
          </div>

          {/* Canvas Area */}
          <div className="flex-1 relative bg-muted/20 overflow-auto">
            {/* Grid Background */}
            <div 
              className="absolute inset-0 opacity-20"
              style={{
                backgroundImage: `
                  linear-gradient(hsl(var(--muted-foreground)) 1px, transparent 1px),
                  linear-gradient(90deg, hsl(var(--muted-foreground)) 1px, transparent 1px)
                `,
                backgroundSize: '20px 20px'
              }}
            />

            {/* Components */}
            <DragDropContext onDragEnd={handleDragEnd}>
              <Droppable droppableId="canvas">
                {(provided) => (
                  <div
                    {...provided.droppableProps}
                    ref={provided.innerRef}
                    className="relative min-h-full p-4"
                  >
                    {components.map((component, index) => (
                      <Draggable key={component.id} draggableId={component.id} index={index}>
                        {(provided, snapshot) => (
                          <div
                            ref={provided.innerRef}
                            {...provided.draggableProps}
                            className={`absolute p-4 bg-card border-2 rounded-lg shadow-sm min-w-48 ${
                              selectedComponent === component.id 
                                ? 'border-primary ring-2 ring-primary/20' 
                                : 'border-border hover:border-primary/50'
                            } ${snapshot.isDragging ? 'shadow-lg' : ''}`}
                            style={{
                              left: component.position.x,
                              top: component.position.y,
                              ...provided.draggableProps.style
                            }}
                            onClick={() => setSelectedComponent(component.id)}
                          >
                            <div {...provided.dragHandleProps} className="flex items-center justify-between mb-2">
                              <div className="flex items-center space-x-2">
                                <Move className="h-4 w-4 text-muted-foreground cursor-grab" />
                                <span className="font-medium text-foreground">{component.name}</span>
                              </div>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation()
                                  removeComponent(component.id)
                                }}
                                className="p-1 rounded hover:bg-destructive/10 text-destructive"
                              >
                                <Trash2 className="h-3 w-3" />
                              </button>
                            </div>
                            
                            <p className="text-xs text-muted-foreground mb-2">{component.description}</p>
                            
                            <div className={`px-2 py-1 rounded-full text-xs font-medium w-fit ${
                              component.type === 'capability' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400' :
                              component.type === 'tool' ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400' :
                              'bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-400'
                            }`}>
                              {component.type}
                            </div>
                          </div>
                        )}
                      </Draggable>
                    ))}
                    {provided.placeholder}
                    
                    {/* Empty State */}
                    {components.length === 0 && (
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center">
                          <Bot className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                          <h3 className="text-lg font-medium text-foreground mb-2">
                            Start Building Your Agent
                          </h3>
                          <p className="text-muted-foreground mb-4">
                            Add components from the library to create your custom agent
                          </p>
                          <button
                            onClick={() => setShowComponentLibrary(true)}
                            className="btn-primary"
                          >
                            <Plus className="h-4 w-4 mr-2" />
                            Add Components
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </Droppable>
            </DragDropContext>
          </div>
        </div>

        {/* Properties Panel */}
        {selectedComponent && (
          <div className="w-80 bg-card border-l border-border p-4">
            <h3 className="text-lg font-semibold text-foreground mb-4">Component Properties</h3>
            {(() => {
              const component = components.find(c => c.id === selectedComponent)
              if (!component) return null

              return (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-foreground mb-2">Name</label>
                    <input
                      type="text"
                      value={component.name}
                      onChange={(e) => {
                        setComponents(prev => prev.map(c => 
                          c.id === selectedComponent ? { ...c, name: e.target.value } : c
                        ))
                      }}
                      className="input"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-foreground mb-2">Description</label>
                    <textarea
                      value={component.description}
                      onChange={(e) => {
                        setComponents(prev => prev.map(c => 
                          c.id === selectedComponent ? { ...c, description: e.target.value } : c
                        ))
                      }}
                      className="input resize-none"
                      rows={3}
                    />
                  </div>

                  {component.type === 'model' && (
                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">
                        Temperature: {component.config.temperature}
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="2"
                        step="0.1"
                        value={component.config.temperature}
                        onChange={(e) => updateComponentConfig(selectedComponent, { 
                          temperature: parseFloat(e.target.value) 
                        })}
                        className="w-full"
                      />
                    </div>
                  )}

                  {component.type === 'capability' && component.config.strength !== undefined && (
                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">
                        Strength: {component.config.strength}
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.1"
                        value={component.config.strength}
                        onChange={(e) => updateComponentConfig(selectedComponent, { 
                          strength: parseFloat(e.target.value) 
                        })}
                        className="w-full"
                      />
                    </div>
                  )}

                  <div className="pt-4 border-t border-border">
                    <h4 className="text-sm font-medium text-foreground mb-2">Configuration</h4>
                    <pre className="text-xs bg-muted p-2 rounded overflow-auto">
                      {JSON.stringify(component.config, null, 2)}
                    </pre>
                  </div>
                </div>
              )
            })()}
          </div>
        )}
      </div>
    </div>
  )
}

export default VisualAgentBuilder
