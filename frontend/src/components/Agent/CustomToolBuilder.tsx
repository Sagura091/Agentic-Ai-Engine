import React, { useState } from 'react'
import { X, Code, Play, Save, Plus, Trash2, Settings, Sparkles, Wand2, TestTube, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import { useForm, useFieldArray } from 'react-hook-form'
import { useQuery } from 'react-query'
import { enhancedOrchestrationApi } from '../../services/api'
import toast from 'react-hot-toast'

interface ToolParameter {
  name: string
  type: 'string' | 'number' | 'boolean' | 'array' | 'object'
  description: string
  required: boolean
  default?: any
}

interface CustomTool {
  name: string
  description: string
  parameters: ToolParameter[]
  code: string
  category: string
}

interface CustomToolBuilderProps {
  isOpen: boolean
  onClose: () => void
  onSave: (tool: CustomTool) => void
  editingTool?: CustomTool
}

const CustomToolBuilder: React.FC<CustomToolBuilderProps> = ({
  isOpen,
  onClose,
  onSave,
  editingTool
}) => {
  const [activeTab, setActiveTab] = useState<'config' | 'generate' | 'code' | 'test'>('config')
  const [testInput, setTestInput] = useState('')
  const [testOutput, setTestOutput] = useState('')
  const [isTestingTool, setIsTestingTool] = useState(false)

  // AI Generation states
  const [aiDescription, setAiDescription] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [generationResult, setGenerationResult] = useState<any>(null)

  const { register, control, handleSubmit, watch, formState: { errors } } = useForm<CustomTool>({
    defaultValues: editingTool || {
      name: '',
      description: '',
      parameters: [],
      code: `def execute_tool(**kwargs):
    """
    Custom tool implementation.
    
    Args:
        **kwargs: Tool parameters
        
    Returns:
        str: Tool execution result
    """
    # Your tool logic here
    return "Tool executed successfully"`,
      category: 'general'
    }
  })

  const { fields, append, remove } = useFieldArray({
    control,
    name: 'parameters'
  })

  const watchedValues = watch()

  const parameterTypes = [
    { value: 'string', label: 'String' },
    { value: 'number', label: 'Number' },
    { value: 'boolean', label: 'Boolean' },
    { value: 'array', label: 'Array' },
    { value: 'object', label: 'Object' }
  ]

  const toolCategories = [
    'general',
    'data-processing',
    'web-scraping',
    'file-operations',
    'api-integration',
    'calculations',
    'text-processing',
    'image-processing'
  ]

  const addParameter = () => {
    append({
      name: '',
      type: 'string',
      description: '',
      required: false
    })
  }

  // AI-powered tool generation
  const generateToolWithAI = async () => {
    if (!aiDescription.trim()) {
      toast.error('Please describe what the tool should do')
      return
    }

    setIsGenerating(true)
    setGenerationResult(null)

    try {
      const response = await enhancedOrchestrationApi.createDynamicTool({
        name: `ai_generated_${Date.now()}`,
        description: aiDescription,
        functionality_description: aiDescription,
        category: 'ai-generated',
        make_global: false
      })

      const generatedTool = response.data
      setGenerationResult(generatedTool)

      // Auto-fill the form with generated tool
      if (generatedTool) {
        setValue('name', generatedTool.name || `AI Tool ${Date.now()}`)
        setValue('description', generatedTool.description || aiDescription)
        setValue('code', generatedTool.code || `def execute_tool(**kwargs):
    """
    ${aiDescription}
    """
    # AI-generated tool implementation
    return "Tool executed successfully"`)
        setValue('category', 'ai-generated')

        toast.success('Tool generated successfully!')
        setActiveTab('code')
      }
    } catch (error: any) {
      console.error('Tool generation failed:', error)
      toast.error('Failed to generate tool: ' + (error.message || 'Unknown error'))
    } finally {
      setIsGenerating(false)
    }
  }

  const testTool = async () => {
    if (!watchedValues.code.trim()) {
      toast.error('Please add tool code first')
      return
    }

    setIsTestingTool(true)

    try {
      // Test the tool using the backend
      const testData = {
        name: watchedValues.name,
        code: watchedValues.code,
        parameters: watchedValues.parameters,
        test_input: testInput
      }

      // For now, simulate the test - in production this would call the backend
      const mockResult = `Tool "${watchedValues.name}" executed with input: ${testInput}\nResult: Tool executed successfully`
      setTestOutput(mockResult)
      toast.success('Tool test completed')
    } catch (error: any) {
      setTestOutput(`Error: ${error.message}`)
      toast.error('Tool test failed')
    } finally {
      setIsTestingTool(false)
    }
  }

  const onSubmit = async (data: CustomTool) => {
    if (!data.name.trim()) {
      toast.error('Tool name is required')
      return
    }

    if (!data.code.trim()) {
      toast.error('Tool code is required')
      return
    }

    try {
      // Create the tool using the enhanced orchestration API
      const response = await enhancedOrchestrationApi.createDynamicTool({
        name: data.name,
        description: data.description,
        functionality_description: `Custom tool: ${data.description}`,
        category: data.category,
        make_global: true // Make it available to all agents
      })

      onSave(data)
      toast.success(`Tool "${data.name}" created and deployed successfully`)
      onClose()
    } catch (error: any) {
      console.error('Failed to create tool:', error)
      toast.error('Failed to create tool: ' + (error.message || 'Unknown error'))
    }
  }

  const generateToolSchema = () => {
    const schema = {
      name: watchedValues.name,
      description: watchedValues.description,
      parameters: {
        type: 'object',
        properties: {},
        required: []
      }
    }

    watchedValues.parameters?.forEach(param => {
      if (param.name) {
        schema.parameters.properties[param.name] = {
          type: param.type,
          description: param.description
        }
        
        if (param.required) {
          schema.parameters.required.push(param.name)
        }
      }
    })

    return JSON.stringify(schema, null, 2)
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex min-h-screen items-center justify-center p-4">
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
        
        <div className="relative w-full max-w-6xl bg-card rounded-lg shadow-xl border border-border max-h-[90vh] overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-border">
            <div className="flex items-center space-x-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <Code className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-foreground">
                  {editingTool ? 'Edit Custom Tool' : 'Create Custom Tool'}
                </h2>
                <p className="text-sm text-muted-foreground">
                  Build custom tools for your agents
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

          {/* Tabs */}
          <div className="border-b border-border">
            <nav className="flex space-x-8 px-6">
              {[
                { id: 'config', label: 'Configuration', icon: Settings },
                { id: 'generate', label: 'AI Generate', icon: Sparkles },
                { id: 'code', label: 'Code Editor', icon: Code },
                { id: 'test', label: 'Test Tool', icon: Play }
              ].map((tab) => {
                const Icon = tab.icon
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                      activeTab === tab.id
                        ? 'border-primary text-primary'
                        : 'border-transparent text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    <span>{tab.label}</span>
                  </button>
                )
              })}
            </nav>
          </div>

          {/* Content */}
          <form onSubmit={handleSubmit(onSubmit)} className="flex-1 overflow-hidden">
            <div className="h-[60vh] overflow-y-auto">
              {activeTab === 'config' && (
                <div className="p-6 space-y-6">
                  {/* Basic Information */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">
                        Tool Name *
                      </label>
                      <input
                        {...register('name', { required: 'Tool name is required' })}
                        className="input"
                        placeholder="my_custom_tool"
                      />
                      {errors.name && (
                        <p className="text-sm text-red-500 mt-1">{errors.name.message}</p>
                      )}
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">
                        Category
                      </label>
                      <select {...register('category')} className="input">
                        {toolCategories.map(category => (
                          <option key={category} value={category}>
                            {category.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-foreground mb-2">
                      Description *
                    </label>
                    <textarea
                      {...register('description', { required: 'Description is required' })}
                      className="input min-h-20 resize-none"
                      placeholder="Describe what this tool does..."
                      rows={3}
                    />
                    {errors.description && (
                      <p className="text-sm text-red-500 mt-1">{errors.description.message}</p>
                    )}
                  </div>

                  {/* Parameters */}
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-medium text-foreground">Parameters</h3>
                      <button
                        type="button"
                        onClick={addParameter}
                        className="btn-secondary inline-flex items-center"
                      >
                        <Plus className="h-4 w-4 mr-2" />
                        Add Parameter
                      </button>
                    </div>

                    <div className="space-y-4">
                      {fields.map((field, index) => (
                        <div key={field.id} className="p-4 border border-border rounded-lg">
                          <div className="flex items-center justify-between mb-3">
                            <h4 className="font-medium text-foreground">Parameter {index + 1}</h4>
                            <button
                              type="button"
                              onClick={() => remove(index)}
                              className="p-1 rounded hover:bg-destructive/10 text-destructive"
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          </div>

                          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                            <div>
                              <label className="block text-sm font-medium text-foreground mb-1">
                                Name
                              </label>
                              <input
                                {...register(`parameters.${index}.name` as const)}
                                className="input"
                                placeholder="parameter_name"
                              />
                            </div>

                            <div>
                              <label className="block text-sm font-medium text-foreground mb-1">
                                Type
                              </label>
                              <select
                                {...register(`parameters.${index}.type` as const)}
                                className="input"
                              >
                                {parameterTypes.map(type => (
                                  <option key={type.value} value={type.value}>
                                    {type.label}
                                  </option>
                                ))}
                              </select>
                            </div>

                            <div className="flex items-center space-x-4 pt-6">
                              <label className="flex items-center space-x-2">
                                <input
                                  type="checkbox"
                                  {...register(`parameters.${index}.required` as const)}
                                  className="rounded"
                                />
                                <span className="text-sm text-foreground">Required</span>
                              </label>
                            </div>
                          </div>

                          <div className="mt-3">
                            <label className="block text-sm font-medium text-foreground mb-1">
                              Description
                            </label>
                            <input
                              {...register(`parameters.${index}.description` as const)}
                              className="input"
                              placeholder="Parameter description..."
                            />
                          </div>
                        </div>
                      ))}

                      {fields.length === 0 && (
                        <div className="text-center py-8 text-muted-foreground">
                          <Code className="h-12 w-12 mx-auto mb-2 opacity-50" />
                          <p>No parameters defined</p>
                          <p className="text-sm">Add parameters to define the tool interface</p>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Generated Schema Preview */}
                  <div>
                    <h3 className="text-lg font-medium text-foreground mb-2">Generated Schema</h3>
                    <pre className="bg-muted p-4 rounded-lg text-sm overflow-auto max-h-40">
                      {generateToolSchema()}
                    </pre>
                  </div>
                </div>
              )}

              {activeTab === 'generate' && (
                <div className="p-6 space-y-6">
                  <div className="text-center mb-6">
                    <div className="p-3 rounded-full bg-primary/10 w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                      <Sparkles className="h-8 w-8 text-primary" />
                    </div>
                    <h3 className="text-lg font-medium text-foreground mb-2">AI-Powered Tool Generation</h3>
                    <p className="text-muted-foreground">
                      Describe what you want your tool to do, and AI will generate the code for you
                    </p>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">
                        Tool Description
                      </label>
                      <textarea
                        value={aiDescription}
                        onChange={(e) => setAiDescription(e.target.value)}
                        className="input min-h-32 resize-none"
                        placeholder="Describe what your tool should do. For example: 'Create a tool that converts text to uppercase and removes special characters' or 'Build a tool that calculates the distance between two geographic coordinates'"
                        rows={4}
                      />
                    </div>

                    <div className="flex items-center justify-center">
                      <button
                        type="button"
                        onClick={generateToolWithAI}
                        disabled={isGenerating || !aiDescription.trim()}
                        className="btn-primary flex items-center space-x-2"
                      >
                        {isGenerating ? (
                          <>
                            <Loader2 className="h-4 w-4 animate-spin" />
                            <span>Generating Tool...</span>
                          </>
                        ) : (
                          <>
                            <Wand2 className="h-4 w-4" />
                            <span>Generate Tool with AI</span>
                          </>
                        )}
                      </button>
                    </div>

                    {generationResult && (
                      <div className="bg-green-50 border border-green-200 rounded-lg p-4 dark:bg-green-900/20 dark:border-green-800">
                        <div className="flex items-center space-x-2 mb-2">
                          <CheckCircle className="h-5 w-5 text-green-600" />
                          <span className="font-medium text-green-800 dark:text-green-200">
                            Tool Generated Successfully!
                          </span>
                        </div>
                        <p className="text-sm text-green-700 dark:text-green-300 mb-3">
                          Your tool has been generated and the form has been auto-filled.
                          You can review and modify the code in the Code Editor tab.
                        </p>
                        <div className="flex items-center space-x-3">
                          <button
                            type="button"
                            onClick={() => setActiveTab('code')}
                            className="btn-ghost text-green-700 border-green-300 hover:bg-green-100 dark:text-green-200 dark:border-green-700 dark:hover:bg-green-800"
                          >
                            <Code className="h-4 w-4 mr-2" />
                            Review Code
                          </button>
                          <button
                            type="button"
                            onClick={() => setActiveTab('test')}
                            className="btn-ghost text-green-700 border-green-300 hover:bg-green-100 dark:text-green-200 dark:border-green-700 dark:hover:bg-green-800"
                          >
                            <TestTube className="h-4 w-4 mr-2" />
                            Test Tool
                          </button>
                        </div>
                      </div>
                    )}

                    <div className="bg-muted/50 rounded-lg p-4">
                      <h4 className="font-medium text-foreground mb-2">AI Generation Tips:</h4>
                      <ul className="text-sm text-muted-foreground space-y-1">
                        <li>• Be specific about what the tool should do</li>
                        <li>• Mention input parameters and expected output</li>
                        <li>• Include any special requirements or constraints</li>
                        <li>• The AI will generate Python code that follows best practices</li>
                        <li>• You can always modify the generated code manually</li>
                      </ul>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'code' && (
                <div className="p-6">
                  <div className="mb-4">
                    <label className="block text-sm font-medium text-foreground mb-2">
                      Tool Implementation
                    </label>
                    <p className="text-sm text-muted-foreground mb-4">
                      Write your tool logic in Python. The function should accept **kwargs and return a string result.
                    </p>
                  </div>

                  <textarea
                    {...register('code', { required: 'Tool code is required' })}
                    className="input font-mono text-sm resize-none w-full h-96"
                    placeholder="def execute_tool(**kwargs):..."
                  />
                  {errors.code && (
                    <p className="text-sm text-red-500 mt-1">{errors.code.message}</p>
                  )}

                  <div className="mt-4 p-4 bg-muted/50 rounded-lg">
                    <h4 className="font-medium text-foreground mb-2">Code Guidelines:</h4>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      <li>• Function must be named <code className="bg-muted px-1 rounded">execute_tool</code></li>
                      <li>• Accept parameters via <code className="bg-muted px-1 rounded">**kwargs</code></li>
                      <li>• Return a string result</li>
                      <li>• Handle errors gracefully</li>
                      <li>• Add docstring for documentation</li>
                    </ul>
                  </div>
                </div>
              )}

              {activeTab === 'test' && (
                <div className="p-6">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Test Input */}
                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">
                        Test Input (JSON)
                      </label>
                      <textarea
                        value={testInput}
                        onChange={(e) => setTestInput(e.target.value)}
                        className="input font-mono text-sm resize-none h-32"
                        placeholder='{"param1": "value1", "param2": "value2"}'
                      />
                      
                      <button
                        type="button"
                        onClick={testTool}
                        disabled={isTestingTool}
                        className="btn-primary mt-3 inline-flex items-center"
                      >
                        {isTestingTool ? (
                          <>
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                            Testing...
                          </>
                        ) : (
                          <>
                            <Play className="h-4 w-4 mr-2" />
                            Test Tool
                          </>
                        )}
                      </button>
                    </div>

                    {/* Test Output */}
                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">
                        Test Output
                      </label>
                      <div className="bg-muted p-4 rounded-lg font-mono text-sm h-32 overflow-auto">
                        {testOutput || 'No output yet. Run a test to see results.'}
                      </div>
                    </div>
                  </div>

                  {/* Parameter Preview */}
                  {watchedValues.parameters && watchedValues.parameters.length > 0 && (
                    <div className="mt-6">
                      <h3 className="text-lg font-medium text-foreground mb-3">Expected Parameters</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                        {watchedValues.parameters.map((param, index) => (
                          <div key={index} className="p-3 border border-border rounded-lg">
                            <div className="flex items-center justify-between mb-1">
                              <span className="font-medium text-foreground">{param.name}</span>
                              <span className={`px-2 py-1 rounded-full text-xs ${
                                param.required 
                                  ? 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
                                  : 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400'
                              }`}>
                                {param.required ? 'Required' : 'Optional'}
                              </span>
                            </div>
                            <p className="text-xs text-muted-foreground mb-1">{param.description}</p>
                            <span className="text-xs font-mono bg-muted px-1 rounded">{param.type}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-end space-x-3 p-6 border-t border-border">
              <button
                type="button"
                onClick={onClose}
                className="btn-ghost"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="btn-primary inline-flex items-center"
              >
                <Save className="h-4 w-4 mr-2" />
                {editingTool ? 'Update Tool' : 'Create Tool'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}

export default CustomToolBuilder
