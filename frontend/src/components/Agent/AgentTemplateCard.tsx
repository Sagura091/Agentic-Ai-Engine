import React from 'react'
import { 
  Bot, 
  MessageCircle, 
  Search, 
  Code, 
  GitBranch, 
  BarChart3, 
  PenTool,
  Sparkles,
  ArrowRight
} from 'lucide-react'
import { AgentTemplate } from '../../contexts/AgentContext'

interface AgentTemplateCardProps {
  template: AgentTemplate
  onUseTemplate: (template: AgentTemplate) => void
}

const iconMap = {
  MessageCircle,
  Search,
  Code,
  GitBranch,
  BarChart3,
  PenTool,
  Bot,
  Sparkles,
}

const AgentTemplateCard: React.FC<AgentTemplateCardProps> = ({ template, onUseTemplate }) => {
  const IconComponent = iconMap[template.icon as keyof typeof iconMap] || Bot

  const getCategoryColor = (category: string) => {
    const colors = {
      'General': 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400',
      'Research': 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400',
      'Development': 'bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-400',
      'Workflow': 'bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-400',
      'Analytics': 'bg-pink-100 text-pink-800 dark:bg-pink-900/20 dark:text-pink-400',
      'Creative': 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900/20 dark:text-indigo-400',
    }
    return colors[category as keyof typeof colors] || 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400'
  }

  return (
    <div className="group card p-6 hover:shadow-lg hover:border-primary/50 transition-all duration-200 cursor-pointer">
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

      <p className="text-muted-foreground text-sm mb-4 line-clamp-2">
        {template.description}
      </p>

      <div className="mb-4">
        <h4 className="text-sm font-medium text-foreground mb-2">Capabilities</h4>
        <div className="flex flex-wrap gap-1">
          {template.capabilities.slice(0, 3).map((capability) => (
            <span
              key={capability}
              className="px-2 py-1 bg-accent text-accent-foreground rounded text-xs"
            >
              {capability}
            </span>
          ))}
          {template.capabilities.length > 3 && (
            <span className="px-2 py-1 bg-muted text-muted-foreground rounded text-xs">
              +{template.capabilities.length - 3} more
            </span>
          )}
        </div>
      </div>

      <button
        onClick={() => onUseTemplate(template)}
        className="w-full flex items-center justify-center space-x-2 py-2 px-4 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors group-hover:shadow-md"
      >
        <span>Use Template</span>
        <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
      </button>
    </div>
  )
}

export default AgentTemplateCard
