import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { MessageCircle, Bot, User, Zap, Brain, Settings, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import LLMProviderSelector from './Agent/LLMProviderSelector';
import { useError } from '../contexts/ErrorContext';
import toast from 'react-hot-toast';

interface ConversationMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

interface LLMConfig {
  provider: string;
  model_id: string;
  model_name?: string;
  temperature: number;
  max_tokens: number;
  top_p?: number;
  top_k?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  api_key?: string;
  base_url?: string;
  organization?: string;
  project?: string;
}

interface AgentConfig {
  name: string;
  description: string;
  capabilities: string[];
  tools: string[];
  model: string;
  model_provider: string;
  temperature: number;
  max_tokens: number;
  system_prompt?: string;
  llm_config?: LLMConfig;
}

interface ConversationalAgentCreatorProps {
  onAgentCreated?: (agentConfig: AgentConfig) => void;
}

export const ConversationalAgentCreator: React.FC<ConversationalAgentCreatorProps> = ({
  onAgentCreated
}) => {
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [conversationStage, setConversationStage] = useState('initial');
  const [suggestedConfig, setSuggestedConfig] = useState<AgentConfig | null>(null);
  const [nextQuestions, setNextQuestions] = useState<string[]>([]);
  const [llmConfig, setLlmConfig] = useState<LLMConfig>({
    provider: 'ollama',
    model_id: 'llama3.2:latest',
    temperature: 0.7,
    max_tokens: 2048
  });
  const [isTestingConfig, setIsTestingConfig] = useState(false);
  const [configTestResult, setConfigTestResult] = useState<any>(null);
  const [isCreatingAgent, setIsCreatingAgent] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { reportError } = useError();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const startConversation = async (userMessage: string) => {
    setIsLoading(true);
    
    try {
      const response = await fetch('/api/v1/conversational-agents/create-agent', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_message: userMessage,
          user_id: 'demo_user'
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to start conversation');
      }

      const data = await response.json();
      
      setSessionId(data.session_id);
      setConversationStage(data.conversation_stage);
      setNextQuestions(data.next_questions || []);
      
      // Add user message
      const userMsg: ConversationMessage = {
        role: 'user',
        content: userMessage,
        timestamp: new Date().toISOString()
      };
      
      // Add assistant response
      const assistantMsg: ConversationMessage = {
        role: 'assistant',
        content: data.assistant_message,
        timestamp: new Date().toISOString()
      };
      
      setMessages([userMsg, assistantMsg]);
      
    } catch (error) {
      console.error('Error starting conversation:', error);
      // Add error message
      const errorMsg: ConversationMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error starting our conversation. Please try again.',
        timestamp: new Date().toISOString()
      };
      setMessages([errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const continueConversation = async (userMessage: string) => {
    if (!sessionId) return;
    
    setIsLoading(true);
    
    try {
      const response = await fetch('/api/v1/conversational-agents/continue-conversation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          user_message: userMessage,
          conversation_history: messages
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to continue conversation');
      }

      const data = await response.json();
      
      setConversationStage(data.conversation_stage);
      setSuggestedConfig(data.suggested_agent_config);
      setNextQuestions(data.next_questions || []);
      
      // Add user message
      const userMsg: ConversationMessage = {
        role: 'user',
        content: userMessage,
        timestamp: new Date().toISOString()
      };
      
      // Add assistant response
      const assistantMsg: ConversationMessage = {
        role: 'assistant',
        content: data.assistant_message,
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, userMsg, assistantMsg]);
      
    } catch (error) {
      console.error('Error continuing conversation:', error);
      const errorMsg: ConversationMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!currentMessage.trim()) return;
    
    const message = currentMessage.trim();
    setCurrentMessage('');
    
    if (messages.length === 0) {
      await startConversation(message);
    } else {
      await continueConversation(message);
    }
  };

  const handleQuestionClick = (question: string) => {
    setCurrentMessage(question);
  };

  const testAgentConfiguration = async () => {
    if (!suggestedConfig) {
      toast.error('No agent configuration to test');
      return;
    }

    setIsTestingConfig(true);
    setConfigTestResult(null);

    try {
      // Test LLM provider connection first
      const providerResponse = await fetch(`/api/v1/llm/test/providers/${llmConfig.provider}`);

      if (!providerResponse.ok) {
        throw new Error(`Failed to test ${llmConfig.provider} provider`);
      }

      const providerTest = await providerResponse.json();

      if (!providerTest.test_result?.is_available) {
        setConfigTestResult({
          success: false,
          error: `${llmConfig.provider} provider is not available: ${providerTest.test_result?.error_message || 'Unknown error'}`,
          provider_test: providerTest.test_result
        });
        return;
      }

      // Test the complete agent configuration
      const testConfig = {
        name: suggestedConfig.name,
        description: suggestedConfig.description,
        model: llmConfig.model_id,
        model_provider: llmConfig.provider,
        temperature: llmConfig.temperature,
        max_tokens: llmConfig.max_tokens,
        capabilities: suggestedConfig.capabilities || [],
        tools: suggestedConfig.tools || [],
        system_prompt: suggestedConfig.system_prompt || `You are ${suggestedConfig.name}. ${suggestedConfig.description}`
      };

      const testResponse = await fetch('/api/v1/agents/test-config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(testConfig),
      });

      if (!testResponse.ok) {
        throw new Error('Failed to test agent configuration');
      }

      const testResult = await testResponse.json();
      setConfigTestResult(testResult);

      if (testResult.success) {
        toast.success('Agent configuration test successful!');
      } else {
        toast.error(`Configuration test failed: ${testResult.error}`);
      }

    } catch (error: any) {
      const errorMessage = error.message || 'Failed to test configuration';
      reportError(error, {
        source: 'ConversationalAgentCreator.testConfiguration',
        type: 'network'
      });

      setConfigTestResult({
        success: false,
        error: errorMessage,
        connectivity_test: false,
        functionality_test: false
      });

      toast.error(errorMessage);
    } finally {
      setIsTestingConfig(false);
    }
  };

  const handleCreateAgent = async () => {
    if (!suggestedConfig) {
      toast.error('No agent configuration available');
      return;
    }

    if (!configTestResult?.success) {
      toast.error('Please test the configuration first');
      return;
    }

    setIsCreatingAgent(true);

    try {
      // Create the agent with the tested configuration
      const agentData = {
        name: suggestedConfig.name,
        description: suggestedConfig.description,
        agent_type: 'conversational',
        model: llmConfig.model_id,
        model_provider: llmConfig.provider,
        temperature: llmConfig.temperature,
        max_tokens: llmConfig.max_tokens,
        capabilities: suggestedConfig.capabilities || [],
        tools: suggestedConfig.tools || [],
        system_prompt: suggestedConfig.system_prompt || `You are ${suggestedConfig.name}. ${suggestedConfig.description}`
      };

      const response = await fetch('/api/v1/agents/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(agentData),
      });

      if (!response.ok) {
        throw new Error('Failed to create agent');
      }

      const createdAgent = await response.json();

      toast.success(`Agent "${suggestedConfig.name}" created successfully!`);

      // Update the config with the created agent ID
      const finalConfig = {
        ...suggestedConfig,
        agent_id: createdAgent.agent_id,
        model: llmConfig.model_id,
        model_provider: llmConfig.provider,
        temperature: llmConfig.temperature,
        max_tokens: llmConfig.max_tokens,
        llm_config: llmConfig
      };

      if (onAgentCreated) {
        onAgentCreated(finalConfig);
      }

    } catch (error: any) {
      const errorMessage = error.message || 'Failed to create agent';
      reportError(error, {
        source: 'ConversationalAgentCreator.createAgent',
        type: 'network'
      });

      toast.error(errorMessage);
    } finally {
      setIsCreatingAgent(false);
    }
  };

  const getStageIcon = () => {
    switch (conversationStage) {
      case 'requirements_gathering':
        return <MessageCircle className="h-4 w-4" />;
      case 'configuration_ready':
        return <Settings className="h-4 w-4" />;
      default:
        return <Bot className="h-4 w-4" />;
    }
  };

  const getStageLabel = () => {
    switch (conversationStage) {
      case 'requirements_gathering':
        return 'Gathering Requirements';
      case 'configuration_ready':
        return 'Configuration Ready';
      default:
        return 'Getting Started';
    }
  };

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto p-4 space-y-4">
      {/* Header */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-blue-500" />
            Conversational Agent Creator
            <Badge variant="outline" className="ml-auto flex items-center gap-1">
              {getStageIcon()}
              {getStageLabel()}
            </Badge>
          </CardTitle>
        </CardHeader>
      </Card>

      {/* Chat Area */}
      <Card className="flex-1 flex flex-col">
        <CardContent className="flex-1 flex flex-col p-4">
          <ScrollArea className="flex-1 pr-4">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center space-y-4">
                <Bot className="h-12 w-12 text-gray-400" />
                <div>
                  <h3 className="text-lg font-semibold text-gray-700">Let's Create Your Agent</h3>
                  <p className="text-gray-500 mt-2">
                    Tell me what you'd like your AI agent to do, and I'll help you design it through conversation.
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex gap-3 ${
                      message.role === 'user' ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    {message.role === 'assistant' && (
                      <div className="flex-shrink-0">
                        <Bot className="h-8 w-8 p-1.5 bg-blue-100 text-blue-600 rounded-full" />
                      </div>
                    )}
                    <div
                      className={`max-w-[80%] p-3 rounded-lg ${
                        message.role === 'user'
                          ? 'bg-blue-500 text-white'
                          : 'bg-gray-100 text-gray-800'
                      }`}
                    >
                      <p className="whitespace-pre-wrap">{message.content}</p>
                      <p className="text-xs opacity-70 mt-1">
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                    {message.role === 'user' && (
                      <div className="flex-shrink-0">
                        <User className="h-8 w-8 p-1.5 bg-gray-100 text-gray-600 rounded-full" />
                      </div>
                    )}
                  </div>
                ))}
                {isLoading && (
                  <div className="flex gap-3 justify-start">
                    <div className="flex-shrink-0">
                      <Bot className="h-8 w-8 p-1.5 bg-blue-100 text-blue-600 rounded-full" />
                    </div>
                    <div className="bg-gray-100 p-3 rounded-lg">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
            <div ref={messagesEndRef} />
          </ScrollArea>

          {/* Suggested Questions */}
          {nextQuestions.length > 0 && (
            <div className="mt-4 space-y-2">
              <Separator />
              <p className="text-sm text-gray-600">Suggested questions:</p>
              <div className="flex flex-wrap gap-2">
                {nextQuestions.map((question, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    size="sm"
                    onClick={() => handleQuestionClick(question)}
                    className="text-left h-auto p-2 whitespace-normal"
                  >
                    {question}
                  </Button>
                ))}
              </div>
            </div>
          )}

          {/* Input Area */}
          <div className="mt-4 space-y-3">
            <Separator />
            <div className="flex gap-2">
              <Input
                value={currentMessage}
                onChange={(e) => setCurrentMessage(e.target.value)}
                placeholder="Describe what you want your agent to do..."
                onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
                disabled={isLoading}
                className="flex-1"
              />
              <Button 
                onClick={handleSendMessage} 
                disabled={!currentMessage.trim() || isLoading}
              >
                Send
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Agent Configuration Preview */}
      {suggestedConfig && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-green-500" />
              Suggested Agent Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm font-medium text-gray-700">Name</p>
                <p className="text-sm text-gray-600">{suggestedConfig.name}</p>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-700">Model</p>
                <p className="text-sm text-gray-600">{suggestedConfig.model}</p>
              </div>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-700">Description</p>
              <p className="text-sm text-gray-600">{suggestedConfig.description}</p>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-700">Capabilities</p>
              <div className="flex flex-wrap gap-1 mt-1">
                {suggestedConfig.capabilities.map((capability, index) => (
                  <Badge key={index} variant="secondary" className="text-xs">
                    {capability}
                  </Badge>
                ))}
              </div>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-700">Tools</p>
              <div className="flex flex-wrap gap-1 mt-1">
                {suggestedConfig.tools.map((tool, index) => (
                  <Badge key={index} variant="outline" className="text-xs">
                    {tool}
                  </Badge>
                ))}
              </div>
            </div>
            <Button onClick={handleCreateAgent} className="w-full">
              Create This Agent
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ConversationalAgentCreator;
