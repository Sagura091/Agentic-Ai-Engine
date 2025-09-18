import React from 'react'
import { 
  FileText, 
  Code, 
  Database, 
  PenTool, 
  Lightbulb, 
  CheckCircle,
  Clock,
  Users,
  Play,
  ArrowRight
} from 'lucide-react'

interface WorkflowTemplate {
  id: string
  name: string
  description: string
  workflow_type: string
  agents: string[]
  steps: Array<{
    id: string
    name: string
    agent: string
  }>
  estimated_time: string
  complexity: string
  icon: string
  category: string
}

interface WorkflowTemplateCardProps {
  template: WorkflowTemplate
  onUseTemplate: (template: WorkflowTemplate) => void
}

const iconMap = {
  FileText,
  Code,
  Database,
  PenTool,
  Lightbulb,
  CheckCircle,
}

const WorkflowTemplateCard: React.FC<WorkflowTemplateCardProps> = ({ template, onUseTemplate }) => {
  const IconComponent = iconMap[template.icon as keyof typeof iconMap] || FileText

  const getComplexityColor = (complexity: string) => {
    switch (complexity.toLowerCase()) {
      case 'low':
        return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400'
      case 'high':
        return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400'
    }
  }

  const getCategoryColor = (category: string) => {
    const colors = {
      'Research': 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400',
      'Development': 'bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-400',
      'Analytics': 'bg-pink-100 text-pink-800 dark:bg-pink-900/20 dark:text-pink-400',
      'Creative': 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900/20 dark:text-indigo-400',
      'Problem Solving': 'bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-400',
      'General': 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400',
    }
    return colors[category as keyof typeof colors] || 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400'
  }

  return (
    <div className="group card p-6 hover:shadow-lg hover:border-primary/50 transition-all duration-200">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors">
            <IconComponent className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground group-hover:text-primary transition-colors">
              {template.name}
            </h3>
            <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${getCategoryColor(template.category)}`}>
              {template.category}
            </span>
          </div>
        </div>
      </div>

      {/* Description */}
      <p className="text-muted-foreground text-sm mb-4 line-clamp-2">
        {template.description}
      </p>

      {/* Workflow Info */}
      <div className="space-y-3 mb-4">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center space-x-2">
            <Users className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Agents:</span>
          </div>
          <span className="font-medium text-foreground">{template.agents.length}</span>
        </div>

        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center space-x-2">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Duration:</span>
          </div>
          <span className="font-medium text-foreground">{template.estimated_time}</span>
        </div>

        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Complexity:</span>
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getComplexityColor(template.complexity)}`}>
            {template.complexity}
          </span>
        </div>
      </div>

      {/* Steps Preview */}
      <div className="mb-4">
        <h4 className="text-sm font-medium text-foreground mb-2">Workflow Steps</h4>
        <div className="space-y-2">
          {template.steps.slice(0, 3).map((step, index) => (
            <div key={step.id} className="flex items-center space-x-2 text-sm">
              <div className="flex-shrink-0 w-5 h-5 bg-primary/10 text-primary rounded-full flex items-center justify-center text-xs font-medium">
                {index + 1}
              </div>
              <span className="text-muted-foreground truncate">{step.name}</span>
            </div>
          ))}
          {template.steps.length > 3 && (
            <div className="flex items-center space-x-2 text-sm">
              <div className="flex-shrink-0 w-5 h-5 bg-muted text-muted-foreground rounded-full flex items-center justify-center text-xs">
                +
              </div>
              <span className="text-muted-foreground">
                {template.steps.length - 3} more steps
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center space-x-2">
        <button
          onClick={() => onUseTemplate(template)}
          className="flex-1 flex items-center justify-center space-x-2 py-2 px-4 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors group-hover:shadow-md"
        >
          <Play className="h-4 w-4" />
          <span>Execute Workflow</span>
        </button>
        
        <button className="p-2 border border-border rounded-md hover:bg-accent transition-colors">
          <ArrowRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  )
}

export default WorkflowTemplateCard
