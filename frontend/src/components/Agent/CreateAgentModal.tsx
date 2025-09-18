import React, { useState } from 'react'
import { useForm } from 'react-hook-form'
import { X, Bot, Sparkles, Settings, Code, TestTube, CheckCircle, AlertCircle, Loader2, Database, Wrench, Brain, Globe, ArrowRight, Plus } from 'lucide-react'
import { useAgent, AgentTemplate } from '../../contexts/AgentContext'
import { enhancedOrchestrationApi, knowledgeBaseApi } from '../../services/api'
import { useQuery } from 'react-query'
import LLMProviderSelector from './LLMProviderSelector'
import { useError } from '../../contexts/ErrorContext'
import { agentApi } from '../../services/api'
import toast from 'react-hot-toast'

interface CreateAgentModalProps {
  isOpen: boolean
  onClose: () => void
  templates: AgentTemplate[]
  selectedTemplate?: AgentTemplate
}

interface LLMConfig {
  provider: string
  model_id: string
  model_name?: string
  temperature: number
  max_tokens: number
  top_p?: number
  top_k?: number
  frequency_penalty?: number
  presence_penalty?: number
  api_key?: string
  base_url?: string
  organization?: string
  project?: string
}

interface AgentFormData {
  name: string
  description: string
  agent_type: string
  system_prompt: string
  capabilities: string[]
  llm_config: LLMConfig
}

interface KnowledgeBaseConfig {
  enabled: boolean
  knowledge_bases: string[]
  create_new: boolean
  new_kb_name?: string
  new_kb_description?: string
}

interface ToolConfig {
  enabled_tools: string[]
  create_custom_tools: boolean
  custom_tools: Array<{
    name: string
    description: string
    functionality: string
  }>
}

interface AutonomousConfig {
  enabled: boolean
  autonomy_level: 'basic' | 'adaptive' | 'advanced'
  learning_mode: 'passive' | 'active' | 'aggressive'
  decision_threshold: number
  enable_proactive_behavior: boolean
  enable_goal_setting: boolean
}

const CreateAgentModal: React.FC<CreateAgentModalProps> = ({
  isOpen,
  onClose,
  templates,
  selectedTemplate
}) => {
  const { createAgent, isLoading } = useAgent()
  const { reportError } = useError()
  const [step, setStep] = useState<number | string>(1)
  const [selectedTemplateLocal, setSelectedTemplateLocal] = useState<AgentTemplate | null>(
    selectedTemplate || null
  )
  const [isTestingConfig, setIsTestingConfig] = useState(false)
  const [configTestResult, setConfigTestResult] = useState<any>(null)
  const [isCreatingAgent, setIsCreatingAgent] = useState(false)

  const { register, handleSubmit, setValue, getValues, formState: { errors } } = useForm<AgentFormData>({
    defaultValues: {
      name: selectedTemplate?.name || '',
      description: selectedTemplate?.description || '',
      agent_type: selectedTemplate?.agent_type || 'general',
      system_prompt: selectedTemplate?.system_prompt || '',
      capabilities: selectedTemplate?.capabilities || [],
      llm_config: {
        provider: 'ollama',
        model_id: '',
        temperature: 0.7,
        max_tokens: 2048,
        base_url: 'http://localhost:11434'
      }
    }
  })

  const [llmConfig, setLlmConfig] = useState<LLMConfig>({
    provider: 'ollama',
    model_id: '',
    temperature: 0.7,
    max_tokens: 2048,
    base_url: 'http://localhost:11434'
  })

  // Enhanced configuration states
  const [knowledgeConfig, setKnowledgeConfig] = useState<KnowledgeBaseConfig>({
    enabled: false,
    knowledge_bases: [],
    create_new: false
  })

  const [toolConfig, setToolConfig] = useState<ToolConfig>({
    enabled_tools: [],
    create_custom_tools: false,
    custom_tools: []
  })

  const [autonomousConfig, setAutonomousConfig] = useState<AutonomousConfig>({
    enabled: false,
    autonomy_level: 'basic',
    learning_mode: 'passive',
    decision_threshold: 0.7,
    enable_proactive_behavior: false,
    enable_goal_setting: false
  })

  // Fetch available knowledge bases
  const { data: knowledgeBases } = useQuery('knowledge-bases',
    () => knowledgeBaseApi.listKnowledgeBases(), {
    enabled: isOpen
  })

  // Fetch available tools
  const { data: availableTools } = useQuery('available-tools',
    () => enhancedOrchestrationApi.listDynamicTools(), {
    enabled: isOpen
  })



  const testAgentConfiguration = async (data: AgentFormData) => {
    setIsTestingConfig(true)
    setConfigTestResult(null)

    try {
      // Test LLM provider connection first
      const providerResponse = await fetch(`/api/v1/llm/test/providers/${llmConfig.provider}`)

      if (!providerResponse.ok) {
        throw new Error(`Failed to test ${llmConfig.provider} provider`)
      }

      const providerTest = await providerResponse.json()

      if (!providerTest.test_result?.is_available) {
        setConfigTestResult({
          success: false,
          error: `${llmConfig.provider} provider is not available: ${providerTest.test_result?.error_message || 'Unknown error'}`,
          provider_test: providerTest.test_result
        })
        toast.error(`${llmConfig.provider} provider is not available`)
        return
      }

      // Test the complete agent configuration
      const testConfig = {
        name: data.name,
        description: data.description,
        agent_type: data.agent_type || 'general',
        model: llmConfig.model_id,
        model_provider: llmConfig.provider,
        temperature: llmConfig.temperature,
        max_tokens: llmConfig.max_tokens,
        capabilities: data.capabilities || [],
        tools: [], // Will be added in future versions
        system_prompt: data.system_prompt || `You are ${data.name}. ${data.description}`
      }

      console.log('🧪 Testing agent configuration:', testConfig)

      // Use the API service that handles authentication
      const testResult = await agentApi.testAgentConfig(testConfig)
      console.log('✅ Test result:', testResult)
      setConfigTestResult(testResult)

      if (testResult.success) {
        toast.success('Agent configuration test successful!')
      } else {
        toast.error(`Configuration test failed: ${testResult.error}`)
      }

    } catch (error: any) {
      const errorMessage = error.message || 'Failed to test configuration'
      reportError(error, {
        source: 'CreateAgentModal.testConfiguration',
        type: 'network'
      })

      setConfigTestResult({
        success: false,
        error: errorMessage,
        connectivity_test: false,
        functionality_test: false
      })

      toast.error(errorMessage)
    } finally {
      setIsTestingConfig(false)
    }
  }

  const onSubmit = async (data: AgentFormData) => {
    // Check if configuration has been tested
    if (!configTestResult?.success) {
      toast.error('Please test the configuration first before creating the agent')
      return
    }

    setIsCreatingAgent(true)

    try {
      // Create new knowledge base if requested
      let selectedKnowledgeBases = knowledgeConfig.knowledge_bases
      if (knowledgeConfig.create_new && knowledgeConfig.new_kb_name) {
        const newKb = await knowledgeBaseApi.createKnowledgeBase({
          name: knowledgeConfig.new_kb_name,
          description: knowledgeConfig.new_kb_description || `Knowledge base for ${data.name}`,
          tags: [data.agent_type, 'agent-specific']
        })
        selectedKnowledgeBases = [...selectedKnowledgeBases, newKb.id]
        toast.success(`Created knowledge base: ${knowledgeConfig.new_kb_name}`)
      }

      // Create custom tools if requested
      const customToolNames: string[] = []
      if (toolConfig.create_custom_tools) {
        for (const tool of toolConfig.custom_tools) {
          if (tool.name && tool.description && tool.functionality) {
            const createdTool = await enhancedOrchestrationApi.createDynamicTool({
              name: tool.name,
              description: tool.description,
              functionality_description: tool.functionality,
              category: 'custom',
              make_global: false
            })
            customToolNames.push(tool.name)
          }
        }
        if (customToolNames.length > 0) {
          toast.success(`Created ${customToolNames.length} custom tools`)
        }
      }

      // Prepare enhanced agent configuration
      const enhancedConfig = {
        llm_config: {
          provider: llmConfig.provider,
          model_id: llmConfig.model_id,
          temperature: llmConfig.temperature,
          max_tokens: llmConfig.max_tokens,
          top_p: llmConfig.top_p,
          top_k: llmConfig.top_k,
          api_key: llmConfig.api_key,
          base_url: llmConfig.base_url
        },
        knowledge_config: knowledgeConfig.enabled ? {
          knowledge_bases: selectedKnowledgeBases,
          enable_rag: true,
          search_strategy: 'hybrid'
        } : undefined,
        autonomous_config: autonomousConfig.enabled ? {
          autonomy_level: autonomousConfig.autonomy_level,
          learning_mode: autonomousConfig.learning_mode,
          decision_threshold: autonomousConfig.decision_threshold,
          enable_proactive_behavior: autonomousConfig.enable_proactive_behavior,
          enable_goal_setting: autonomousConfig.enable_goal_setting
        } : undefined
      }

      // Create the agent with enhanced orchestration
      const agentData = {
        agent_type: data.agent_type as 'basic' | 'autonomous' | 'research' | 'creative' | 'optimization' | 'custom',
        name: data.name,
        description: data.description,
        tools: [...toolConfig.enabled_tools, ...customToolNames],
        config: enhancedConfig
      }

      const response = await enhancedOrchestrationApi.createAgent(agentData)

      // If autonomous mode is enabled, create autonomous agent configuration
      if (autonomousConfig.enabled && response.data?.agent_id) {
        try {
          // Note: Autonomous capabilities are configured in the agent config
          // The backend will handle autonomous execution setup automatically
          toast.success('Agent configured for autonomous execution!')
        } catch (autonomousError) {
          console.warn('Failed to configure autonomous execution:', autonomousError)
          // Don't fail the entire creation for this
        }
      }

      toast.success('Enhanced agent created successfully!')
      onClose()

      // Reset form state
      setStep(1)
      setConfigTestResult(null)
      setKnowledgeConfig({ enabled: false, knowledge_bases: [], create_new: false })
      setToolConfig({ enabled_tools: [], create_custom_tools: false, custom_tools: [] })
      setAutonomousConfig({
        enabled: false,
        autonomy_level: 'basic',
        learning_mode: 'passive',
        decision_threshold: 0.7,
        enable_proactive_behavior: false,
        enable_goal_setting: false
      })

    } catch (error: any) {
      const errorMessage = error.message || 'Failed to create agent'
      reportError(error, {
        source: 'CreateAgentModal.createAgent',
        type: 'network'
      })
      toast.error(errorMessage)
    } finally {
      setIsCreatingAgent(false)
    }
  }

  const handleTemplateSelect = (template: AgentTemplate) => {
    setSelectedTemplateLocal(template)
    setValue('name', template.name)
    setValue('description', template.description)
    setValue('agent_type', template.agent_type)
    setValue('system_prompt', template.system_prompt)
    setValue('capabilities', template.capabilities)
    setStep(2)
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex min-h-screen items-center justify-center p-4">
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
        
        <div className="relative w-full max-w-2xl bg-card rounded-lg shadow-xl border border-border">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-border">
            <div className="flex items-center space-x-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <Bot className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-foreground">Create New Agent</h2>
                <p className="text-sm text-muted-foreground">
                  Step {step} of 2: {step === 1 ? 'Choose Template' : 'Configure Agent'}
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-md hover:bg-accent transition-colors"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Content */}
          <div className="p-6">
            {step === 1 ? (
              /* Step 1: Template Selection */
              <div className="space-y-4">
                <div className="text-center mb-6">
                  <h3 className="text-lg font-medium text-foreground mb-2">Choose a Template</h3>
                  <p className="text-muted-foreground">
                    Start with a pre-configured template or create from scratch
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-96 overflow-y-auto custom-scrollbar">
                  {/* Custom Agent Option */}
                  <button
                    onClick={() => setStep(2)}
                    className="p-4 border-2 border-dashed border-border hover:border-primary rounded-lg transition-colors text-left"
                  >
                    <div className="flex items-center space-x-3 mb-2">
                      <div className="p-2 rounded-lg bg-muted">
                        <Settings className="h-5 w-5 text-muted-foreground" />
                      </div>
                      <h4 className="font-medium text-foreground">Custom Agent</h4>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Create a completely custom agent from scratch
                    </p>
                  </button>

                  {/* Template Options */}
                  {templates.map((template) => (
                    <button
                      key={template.id}
                      onClick={() => handleTemplateSelect(template)}
                      className="p-4 border border-border hover:border-primary rounded-lg transition-colors text-left"
                    >
                      <div className="flex items-center space-x-3 mb-2">
                        <div className="p-2 rounded-lg bg-primary/10">
                          <Sparkles className="h-5 w-5 text-primary" />
                        </div>
                        <h4 className="font-medium text-foreground">{template.name}</h4>
                      </div>
                      <p className="text-sm text-muted-foreground line-clamp-2">
                        {template.description}
                      </p>
                      <div className="flex flex-wrap gap-1 mt-2">
                        {template.capabilities.slice(0, 2).map((cap) => (
                          <span
                            key={cap}
                            className="px-2 py-1 bg-accent text-accent-foreground rounded text-xs"
                          >
                            {cap}
                          </span>
                        ))}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              /* Step 2: Enhanced Agent Configuration */
              <div className="space-y-6">
                {/* Enhanced Tab Navigation */}
                <div className="border-b border-border">
                  <nav className="flex space-x-8">
                    {[
                      { id: 'basic', label: 'Basic Info', icon: Bot },
                      { id: 'knowledge', label: 'Knowledge Base', icon: Database },
                      { id: 'tools', label: 'Tools', icon: Wrench },
                      { id: 'autonomous', label: 'Autonomous', icon: Brain },
                      { id: 'test', label: 'Test & Deploy', icon: TestTube }
                    ].map((tab) => {
                      const Icon = tab.icon
                      return (
                        <button
                          key={tab.id}
                          type="button"
                          onClick={() => setStep(tab.id as any)}
                          className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                            step === tab.id
                              ? 'border-primary text-primary'
                              : 'border-transparent text-muted-foreground hover:text-foreground hover:border-border'
                          }`}
                        >
                          <Icon className="h-4 w-4" />
                          <span>{tab.label}</span>
                        </button>
                      )
                    })}
                  </nav>
                </div>

                <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                  {/* Basic Information Tab */}
                  {step === 'basic' && (
                    <div className="space-y-4">
                      <h3 className="text-lg font-medium text-foreground flex items-center space-x-2">
                        <Bot className="h-5 w-5" />
                        <span>Basic Information</span>
                      </h3>
                  
                  <div>
                    <label className="block text-sm font-medium text-foreground mb-2">
                      Agent Name *
                    </label>
                    <input
                      {...register('name', { required: 'Agent name is required' })}
                      className="input"
                      placeholder="Enter agent name"
                    />
                    {errors.name && (
                      <p className="text-sm text-red-500 mt-1">{errors.name.message}</p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-foreground mb-2">
                      Description *
                    </label>
                    <textarea
                      {...register('description', { required: 'Description is required' })}
                      className="input min-h-20 resize-none"
                      placeholder="Describe what this agent does"
                      rows={3}
                    />
                    {errors.description && (
                      <p className="text-sm text-red-500 mt-1">{errors.description.message}</p>
                    )}
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">
                        Agent Type
                      </label>
                      <select {...register('agent_type')} className="input">
                        <option value="general">General</option>
                        <option value="research">Research</option>
                        <option value="workflow">Workflow</option>
                      </select>
                    </div>
                  </div>

                  {/* LLM Configuration */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-medium text-foreground">LLM Configuration</h3>
                    <LLMProviderSelector
                      value={llmConfig}
                      onChange={setLlmConfig}
                    />
                  </div>

                  {/* System Prompt */}
                  <div>
                    <label className="block text-sm font-medium text-foreground mb-2">
                      System Prompt
                    </label>
                    <textarea
                      {...register('system_prompt')}
                      className="input min-h-24 resize-none font-mono text-sm"
                      placeholder="Enter system prompt to define agent behavior"
                      rows={4}
                    />
                  </div>
                </div>
                  )}

                  {/* Knowledge Base Configuration Tab */}
                  {step === 'knowledge' && (
                    <div className="space-y-4">
                      <h3 className="text-lg font-medium text-foreground flex items-center space-x-2">
                        <Database className="h-5 w-5" />
                        <span>Knowledge Base Configuration</span>
                      </h3>

                      <div className="space-y-4">
                        <div className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            id="enable-knowledge"
                            checked={knowledgeConfig.enabled}
                            onChange={(e) => setKnowledgeConfig(prev => ({ ...prev, enabled: e.target.checked }))}
                            className="rounded border-border"
                          />
                          <label htmlFor="enable-knowledge" className="text-sm font-medium text-foreground">
                            Enable Knowledge Base Integration
                          </label>
                        </div>

                        {knowledgeConfig.enabled && (
                          <div className="space-y-4 pl-6 border-l-2 border-primary/20">
                            <div>
                              <label className="block text-sm font-medium text-foreground mb-2">
                                Select Existing Knowledge Bases
                              </label>
                              <div className="space-y-2 max-h-32 overflow-y-auto">
                                {knowledgeBases?.map((kb: any) => (
                                  <div key={kb.id} className="flex items-center space-x-2">
                                    <input
                                      type="checkbox"
                                      id={`kb-${kb.id}`}
                                      checked={knowledgeConfig.knowledge_bases.includes(kb.id)}
                                      onChange={(e) => {
                                        if (e.target.checked) {
                                          setKnowledgeConfig(prev => ({
                                            ...prev,
                                            knowledge_bases: [...prev.knowledge_bases, kb.id]
                                          }))
                                        } else {
                                          setKnowledgeConfig(prev => ({
                                            ...prev,
                                            knowledge_bases: prev.knowledge_bases.filter(id => id !== kb.id)
                                          }))
                                        }
                                      }}
                                      className="rounded border-border"
                                    />
                                    <label htmlFor={`kb-${kb.id}`} className="text-sm text-foreground">
                                      {kb.name} ({kb.document_count || 0} documents)
                                    </label>
                                  </div>
                                ))}
                              </div>
                            </div>

                            <div className="border-t border-border pt-4">
                              <div className="flex items-center space-x-2 mb-4">
                                <input
                                  type="checkbox"
                                  id="create-new-kb"
                                  checked={knowledgeConfig.create_new}
                                  onChange={(e) => setKnowledgeConfig(prev => ({ ...prev, create_new: e.target.checked }))}
                                  className="rounded border-border"
                                />
                                <label htmlFor="create-new-kb" className="text-sm font-medium text-foreground">
                                  Create New Knowledge Base
                                </label>
                              </div>

                              {knowledgeConfig.create_new && (
                                <div className="space-y-3 pl-6">
                                  <div>
                                    <label className="block text-sm font-medium text-foreground mb-1">
                                      Knowledge Base Name
                                    </label>
                                    <input
                                      type="text"
                                      value={knowledgeConfig.new_kb_name || ''}
                                      onChange={(e) => setKnowledgeConfig(prev => ({ ...prev, new_kb_name: e.target.value }))}
                                      className="input"
                                      placeholder="Enter knowledge base name"
                                    />
                                  </div>
                                  <div>
                                    <label className="block text-sm font-medium text-foreground mb-1">
                                      Description
                                    </label>
                                    <textarea
                                      value={knowledgeConfig.new_kb_description || ''}
                                      onChange={(e) => setKnowledgeConfig(prev => ({ ...prev, new_kb_description: e.target.value }))}
                                      className="input min-h-16 resize-none"
                                      placeholder="Describe the knowledge base purpose"
                                      rows={2}
                                    />
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Tools Configuration Tab */}
                  {step === 'tools' && (
                    <div className="space-y-4">
                      <h3 className="text-lg font-medium text-foreground flex items-center space-x-2">
                        <Wrench className="h-5 w-5" />
                        <span>Tools Configuration</span>
                      </h3>

                      <div className="space-y-4">
                        <div>
                          <label className="block text-sm font-medium text-foreground mb-2">
                            Available Tools
                          </label>
                          <div className="grid grid-cols-2 gap-2 max-h-40 overflow-y-auto border border-border rounded-lg p-3">
                            {availableTools?.tools?.map((tool: any) => (
                              <div key={tool.name} className="flex items-center space-x-2">
                                <input
                                  type="checkbox"
                                  id={`tool-${tool.name}`}
                                  checked={toolConfig.enabled_tools.includes(tool.name)}
                                  onChange={(e) => {
                                    if (e.target.checked) {
                                      setToolConfig(prev => ({
                                        ...prev,
                                        enabled_tools: [...prev.enabled_tools, tool.name]
                                      }))
                                    } else {
                                      setToolConfig(prev => ({
                                        ...prev,
                                        enabled_tools: prev.enabled_tools.filter(name => name !== tool.name)
                                      }))
                                    }
                                  }}
                                  className="rounded border-border"
                                />
                                <label htmlFor={`tool-${tool.name}`} className="text-sm text-foreground">
                                  {tool.name}
                                </label>
                              </div>
                            ))}
                          </div>
                        </div>

                        <div className="border-t border-border pt-4">
                          <div className="flex items-center space-x-2 mb-4">
                            <input
                              type="checkbox"
                              id="create-custom-tools"
                              checked={toolConfig.create_custom_tools}
                              onChange={(e) => setToolConfig(prev => ({ ...prev, create_custom_tools: e.target.checked }))}
                              className="rounded border-border"
                            />
                            <label htmlFor="create-custom-tools" className="text-sm font-medium text-foreground">
                              Create Custom Tools
                            </label>
                          </div>

                          {toolConfig.create_custom_tools && (
                            <div className="space-y-3 pl-6">
                              {toolConfig.custom_tools.map((tool, index) => (
                                <div key={index} className="border border-border rounded-lg p-3 space-y-2">
                                  <div className="flex items-center justify-between">
                                    <h4 className="text-sm font-medium text-foreground">Custom Tool {index + 1}</h4>
                                    <button
                                      type="button"
                                      onClick={() => {
                                        setToolConfig(prev => ({
                                          ...prev,
                                          custom_tools: prev.custom_tools.filter((_, i) => i !== index)
                                        }))
                                      }}
                                      className="text-red-500 hover:text-red-700"
                                    >
                                      <X className="h-4 w-4" />
                                    </button>
                                  </div>
                                  <input
                                    type="text"
                                    value={tool.name}
                                    onChange={(e) => {
                                      const newTools = [...toolConfig.custom_tools]
                                      newTools[index] = { ...tool, name: e.target.value }
                                      setToolConfig(prev => ({ ...prev, custom_tools: newTools }))
                                    }}
                                    className="input"
                                    placeholder="Tool name"
                                  />
                                  <input
                                    type="text"
                                    value={tool.description}
                                    onChange={(e) => {
                                      const newTools = [...toolConfig.custom_tools]
                                      newTools[index] = { ...tool, description: e.target.value }
                                      setToolConfig(prev => ({ ...prev, custom_tools: newTools }))
                                    }}
                                    className="input"
                                    placeholder="Tool description"
                                  />
                                  <textarea
                                    value={tool.functionality}
                                    onChange={(e) => {
                                      const newTools = [...toolConfig.custom_tools]
                                      newTools[index] = { ...tool, functionality: e.target.value }
                                      setToolConfig(prev => ({ ...prev, custom_tools: newTools }))
                                    }}
                                    className="input min-h-16 resize-none"
                                    placeholder="Describe what this tool should do"
                                    rows={2}
                                  />
                                </div>
                              ))}
                              <button
                                type="button"
                                onClick={() => {
                                  setToolConfig(prev => ({
                                    ...prev,
                                    custom_tools: [...prev.custom_tools, { name: '', description: '', functionality: '' }]
                                  }))
                                }}
                                className="btn-ghost w-full"
                              >
                                <Plus className="h-4 w-4 mr-2" />
                                Add Custom Tool
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Autonomous Configuration Tab */}
                  {step === 'autonomous' && (
                    <div className="space-y-4">
                      <h3 className="text-lg font-medium text-foreground flex items-center space-x-2">
                        <Brain className="h-5 w-5" />
                        <span>Autonomous Configuration</span>
                      </h3>

                      <div className="space-y-4">
                        <div className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            id="enable-autonomous"
                            checked={autonomousConfig.enabled}
                            onChange={(e) => setAutonomousConfig(prev => ({ ...prev, enabled: e.target.checked }))}
                            className="rounded border-border"
                          />
                          <label htmlFor="enable-autonomous" className="text-sm font-medium text-foreground">
                            Enable Autonomous Behavior
                          </label>
                        </div>

                        {autonomousConfig.enabled && (
                          <div className="space-y-4 pl-6 border-l-2 border-primary/20">
                            <div className="grid grid-cols-2 gap-4">
                              <div>
                                <label className="block text-sm font-medium text-foreground mb-2">
                                  Autonomy Level
                                </label>
                                <select
                                  value={autonomousConfig.autonomy_level}
                                  onChange={(e) => setAutonomousConfig(prev => ({
                                    ...prev,
                                    autonomy_level: e.target.value as 'basic' | 'adaptive' | 'advanced'
                                  }))}
                                  className="input"
                                >
                                  <option value="basic">Basic - Follow instructions</option>
                                  <option value="adaptive">Adaptive - Learn and adapt</option>
                                  <option value="advanced">Advanced - Proactive behavior</option>
                                </select>
                              </div>

                              <div>
                                <label className="block text-sm font-medium text-foreground mb-2">
                                  Learning Mode
                                </label>
                                <select
                                  value={autonomousConfig.learning_mode}
                                  onChange={(e) => setAutonomousConfig(prev => ({
                                    ...prev,
                                    learning_mode: e.target.value as 'passive' | 'active' | 'aggressive'
                                  }))}
                                  className="input"
                                >
                                  <option value="passive">Passive - Learn from interactions</option>
                                  <option value="active">Active - Seek learning opportunities</option>
                                  <option value="aggressive">Aggressive - Continuous learning</option>
                                </select>
                              </div>
                            </div>

                            <div>
                              <label className="block text-sm font-medium text-foreground mb-2">
                                Decision Threshold: {autonomousConfig.decision_threshold}
                              </label>
                              <input
                                type="range"
                                min="0.1"
                                max="1.0"
                                step="0.1"
                                value={autonomousConfig.decision_threshold}
                                onChange={(e) => setAutonomousConfig(prev => ({
                                  ...prev,
                                  decision_threshold: parseFloat(e.target.value)
                                }))}
                                className="w-full"
                              />
                              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                                <span>Conservative</span>
                                <span>Balanced</span>
                                <span>Aggressive</span>
                              </div>
                            </div>

                            <div className="space-y-2">
                              <div className="flex items-center space-x-2">
                                <input
                                  type="checkbox"
                                  id="enable-proactive"
                                  checked={autonomousConfig.enable_proactive_behavior}
                                  onChange={(e) => setAutonomousConfig(prev => ({
                                    ...prev,
                                    enable_proactive_behavior: e.target.checked
                                  }))}
                                  className="rounded border-border"
                                />
                                <label htmlFor="enable-proactive" className="text-sm text-foreground">
                                  Enable Proactive Behavior
                                </label>
                              </div>

                              <div className="flex items-center space-x-2">
                                <input
                                  type="checkbox"
                                  id="enable-goal-setting"
                                  checked={autonomousConfig.enable_goal_setting}
                                  onChange={(e) => setAutonomousConfig(prev => ({
                                    ...prev,
                                    enable_goal_setting: e.target.checked
                                  }))}
                                  className="rounded border-border"
                                />
                                <label htmlFor="enable-goal-setting" className="text-sm text-foreground">
                                  Enable Autonomous Goal Setting
                                </label>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Test & Deploy Tab */}
                  {step === 'test' && (
                    <div className="space-y-4">
                      <h3 className="text-lg font-medium text-foreground flex items-center space-x-2">
                        <TestTube className="h-5 w-5" />
                        <span>Test & Deploy</span>
                      </h3>

                      <div className="space-y-4">
                        <div className="bg-muted/50 rounded-lg p-4">
                          <h4 className="font-medium text-foreground mb-2">Configuration Summary</h4>
                          <div className="space-y-1 text-sm text-muted-foreground">
                            <p>• LLM: {llmConfig.provider} - {llmConfig.model_id}</p>
                            <p>• Knowledge Bases: {knowledgeConfig.enabled ? knowledgeConfig.knowledge_bases.length + (knowledgeConfig.create_new ? 1 : 0) : 0}</p>
                            <p>• Tools: {toolConfig.enabled_tools.length + toolConfig.custom_tools.length}</p>
                            <p>• Autonomous: {autonomousConfig.enabled ? 'Enabled' : 'Disabled'}</p>
                          </div>
                        </div>

                        <div className="space-y-3">
                          <button
                            type="button"
                            onClick={() => testAgentConfiguration(getValues())}
                            disabled={isTestingConfig}
                            className="btn-primary w-full"
                          >
                            {isTestingConfig ? (
                              <>
                                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                Testing Configuration...
                              </>
                            ) : (
                              <>
                                <TestTube className="h-4 w-4 mr-2" />
                                Test Configuration
                              </>
                            )}
                          </button>

                          {configTestResult && (
                            <div className={`p-4 rounded-lg border ${
                              configTestResult.success
                                ? 'bg-green-50 border-green-200 text-green-800 dark:bg-green-900/20 dark:border-green-800 dark:text-green-200'
                                : 'bg-red-50 border-red-200 text-red-800 dark:bg-red-900/20 dark:border-red-800 dark:text-red-200'
                            }`}>
                              <div className="flex items-center space-x-2">
                                {configTestResult.success ? (
                                  <CheckCircle className="h-5 w-5" />
                                ) : (
                                  <AlertCircle className="h-5 w-5" />
                                )}
                                <span className="font-medium">
                                  {configTestResult.success ? 'Configuration Valid' : 'Configuration Issues'}
                                </span>
                              </div>
                              {configTestResult.error && (
                                <p className="mt-2 text-sm">{configTestResult.error}</p>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Navigation */}
                  <div className="flex items-center justify-between pt-6 border-t border-border">
                    <button
                      type="button"
                      onClick={() => setStep(1)}
                      className="btn-ghost"
                    >
                      Back to Templates
                    </button>

                    <div className="flex items-center space-x-3">
                      {step !== 'test' && (
                        <button
                          type="button"
                          onClick={() => {
                            const tabs = ['basic', 'knowledge', 'tools', 'autonomous', 'test']
                            const currentIndex = tabs.indexOf(step as string)
                            if (currentIndex < tabs.length - 1) {
                              setStep(tabs[currentIndex + 1])
                            }
                          }}
                          className="btn-ghost"
                        >
                          Next
                          <ArrowRight className="h-4 w-4 ml-2" />
                        </button>
                      )}

                      {step === 'test' && (
                        <button
                          type="submit"
                          disabled={isCreatingAgent || !configTestResult?.success}
                          className="btn-primary"
                        >
                          {isCreatingAgent ? (
                            <>
                              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                              Creating Agent...
                            </>
                          ) : (
                            <>
                              <Bot className="h-4 w-4 mr-2" />
                              Create Agent
                            </>
                          )}
                        </button>
                      )}
                    </div>
                  </div>
                </form>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default CreateAgentModal
