import React, { useState } from 'react'
import { useForm } from 'react-hook-form'
import { X, GitBranch, Play } from 'lucide-react'
import { workflowApi } from '../../services/api'
import { useSocket } from '../../contexts/SocketContext'
import toast from 'react-hot-toast'

interface CreateWorkflowModalProps {
  isOpen: boolean
  onClose: () => void
  templates: any[]
}

interface WorkflowFormData {
  task: string
  workflow_type: string
  model: string
  timeout: number
}

const CreateWorkflowModal: React.FC<CreateWorkflowModalProps> = ({
  isOpen,
  onClose,
  templates
}) => {
  const [isExecuting, setIsExecuting] = useState(false)
  const { executeWorkflow } = useSocket()

  const { register, handleSubmit, watch, formState: { errors } } = useForm<WorkflowFormData>({
    defaultValues: {
      task: '',
      workflow_type: 'hierarchical',
      model: 'llama3.2:latest',
      timeout: 300
    }
  })

  const watchedValues = watch()

  const onSubmit = async (data: WorkflowFormData) => {
    try {
      setIsExecuting(true)
      
      // Execute workflow via WebSocket for real-time updates
      executeWorkflow({
        workflowId: `workflow_${Date.now()}`,
        task: data.task,
        options: {
          workflowType: data.workflow_type,
          model: data.model,
          timeout: data.timeout
        }
      })

      toast.success('Workflow execution started!')
      onClose()
    } catch (error: any) {
      toast.error(error.message || 'Failed to execute workflow')
    } finally {
      setIsExecuting(false)
    }
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
                <GitBranch className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-foreground">Execute Workflow</h2>
                <p className="text-sm text-muted-foreground">
                  Configure and run a multi-agent workflow
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
          <form onSubmit={handleSubmit(onSubmit)} className="p-6 space-y-6">
            {/* Task Description */}
            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Task Description *
              </label>
              <textarea
                {...register('task', { required: 'Task description is required' })}
                className="input min-h-24 resize-none"
                placeholder="Describe the task you want the workflow to accomplish..."
                rows={4}
              />
              {errors.task && (
                <p className="text-sm text-red-500 mt-1">{errors.task.message}</p>
              )}
              <p className="text-xs text-muted-foreground mt-1">
                Be specific about what you want to achieve. The workflow will coordinate multiple agents to complete this task.
              </p>
            </div>

            {/* Workflow Configuration */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Workflow Type
                </label>
                <select {...register('workflow_type')} className="input">
                  <option value="hierarchical">Hierarchical (Recommended)</option>
                  <option value="default_multi_agent">Multi-Agent</option>
                  <option value="sequential">Sequential</option>
                </select>
                <p className="text-xs text-muted-foreground mt-1">
                  Hierarchical workflows use supervisor agents to coordinate specialized teams
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Model
                </label>
                <select {...register('model')} className="input">
                  <option value="llama3.2:latest">Llama 3.2 (Latest)</option>
                  <option value="llama3.1:latest">Llama 3.1</option>
                  <option value="qwen2.5:latest">Qwen 2.5</option>
                  <option value="mistral:latest">Mistral</option>
                </select>
              </div>
            </div>

            {/* Advanced Settings */}
            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Timeout (seconds): {watchedValues.timeout}
              </label>
              <input
                {...register('timeout', { 
                  min: 60, 
                  max: 1800,
                  valueAsNumber: true 
                })}
                type="range"
                min="60"
                max="1800"
                step="30"
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                <span>1 min</span>
                <span>30 min</span>
              </div>
            </div>

            {/* Workflow Preview */}
            <div className="bg-muted/50 rounded-lg p-4">
              <h3 className="text-sm font-medium text-foreground mb-2">Workflow Preview</h3>
              <div className="text-sm text-muted-foreground space-y-1">
                <p>• <strong>Type:</strong> {watchedValues.workflow_type}</p>
                <p>• <strong>Model:</strong> {watchedValues.model}</p>
                <p>• <strong>Estimated Duration:</strong> 3-10 minutes</p>
                <p>• <strong>Agents:</strong> Research Specialist, Data Analyst, Creative Writer</p>
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center justify-end space-x-3 pt-4 border-t border-border">
              <button
                type="button"
                onClick={onClose}
                className="btn-ghost"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isExecuting}
                className="btn-primary inline-flex items-center"
              >
                {isExecuting ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Executing...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Execute Workflow
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}

export default CreateWorkflowModal
