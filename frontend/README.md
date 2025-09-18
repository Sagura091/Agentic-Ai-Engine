# Agentic AI Frontend

A comprehensive React-based frontend application for the standalone Agentic AI microservice, providing visual agent creation and workflow management capabilities.

## 🚀 Features

### Core Functionality
- **Visual Agent Builder**: Create and configure AI agents with intuitive forms
- **Workflow Designer**: Visual drag-and-drop workflow creation with LangGraph integration
- **Real-time Chat**: Interactive chat interface with individual agents
- **Live Monitoring**: Real-time system monitoring and performance metrics
- **Settings Management**: Comprehensive configuration and customization options

### Technical Highlights
- **Modern React**: Built with React 18, TypeScript, and Vite
- **Responsive Design**: Tailwind CSS with dark/light theme support
- **Real-time Communication**: WebSocket integration with Socket.io
- **State Management**: React Query for server state, Context API for client state
- **Production Ready**: Docker containerization and health checks

## 🏗️ Architecture

```
frontend/
├── client/                 # React application
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── contexts/       # React contexts (Theme, Socket, Agent)
│   │   ├── pages/          # Main application pages
│   │   ├── services/       # API service layer
│   │   └── utils/          # Utility functions
│   ├── public/             # Static assets
│   └── dist/               # Built application
├── server/                 # Express.js backend
│   ├── routes/             # API routes
│   └── index.js            # Server entry point
└── docs/                   # Documentation
```

## 🛠️ Technology Stack

### Frontend (Client)
- **React 18** - Modern React with hooks and concurrent features
- **TypeScript** - Type safety and better developer experience
- **Vite** - Fast build tool and development server
- **Tailwind CSS** - Utility-first CSS framework
- **React Query** - Server state management and caching
- **React Router** - Client-side routing
- **React Hook Form** - Form management with validation
- **Socket.io Client** - Real-time WebSocket communication
- **Recharts** - Data visualization and charts
- **Lucide React** - Modern icon library

### Backend (Server)
- **Node.js** - JavaScript runtime
- **Express.js** - Web application framework
- **Socket.io** - Real-time bidirectional communication
- **Axios** - HTTP client for API requests
- **CORS** - Cross-origin resource sharing

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Docker and Docker Compose (for containerized deployment)
- Running Agentic AI backend service

### Development Setup

1. **Clone and navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   # Install server dependencies
   npm install
   
   # Install client dependencies
   cd client && npm install && cd ..
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start development servers**
   ```bash
   # Start both client and server in development mode
   npm run dev
   ```

5. **Access the application**
   - Frontend: http://localhost:5173
   - API Server: http://localhost:3001

### Production Deployment

#### Using Docker Compose

1. **Build and start services**
   ```bash
   docker-compose up -d
   ```

2. **Access the application**
   - Frontend: http://localhost:3001

#### Manual Deployment

1. **Build the client**
   ```bash
   cd client && npm run build && cd ..
   ```

2. **Start the server**
   ```bash
   npm start
   ```

## 📱 Application Pages

### Dashboard
- System overview with key metrics
- Quick actions for common tasks
- Recent activity feed
- Connection status indicators

### Agent Builder
- Browse agent templates by category
- Create new agents with custom configurations
- Manage existing agents
- Real-time agent status monitoring

### Workflow Designer
- Visual workflow canvas with drag-and-drop interface
- Pre-built workflow templates
- Custom workflow creation
- Workflow execution history

### Agent Chat
- Interactive chat interface with agents
- Real-time message streaming
- Conversation history
- Agent performance metrics

### Monitoring
- Real-time system health monitoring
- Performance charts and metrics
- Service status dashboard
- Activity logs and alerts

### Settings
- Theme and appearance customization
- Performance and security configuration
- Data management and export/import
- System preferences

## 🔧 Configuration

### Environment Variables

```bash
# Server Configuration
PORT=3001
NODE_ENV=development

# Client Configuration
CLIENT_URL=http://localhost:5173
VITE_API_URL=http://localhost:3001/api
VITE_SERVER_URL=http://localhost:3001
VITE_WS_URL=ws://localhost:3001

# Agentic AI Service
AGENTIC_API_URL=http://localhost:8001

# Security
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Logging
LOG_LEVEL=info
```

### API Integration

The frontend communicates with the Agentic AI backend through:
- **REST API**: Standard HTTP requests for CRUD operations
- **WebSocket**: Real-time updates for agent execution and monitoring
- **Proxy Routes**: Server-side proxy to the Agentic AI service

## 🎨 Theming and Customization

### Theme System
- **Light/Dark/System** themes with automatic switching
- **CSS Custom Properties** for consistent design tokens
- **Tailwind CSS** configuration with custom color palette
- **Component-level** styling with utility classes

### Customization Options
- Color schemes and accent colors
- Typography and spacing
- Animation preferences
- Layout density (compact/comfortable)

## 🔌 API Integration

### Service Layer Architecture
```typescript
// API service structure
services/
├── api.ts              # Base API configuration
├── agents.ts           # Agent management
├── workflows.ts        # Workflow operations
├── monitoring.ts       # System monitoring
└── settings.ts         # Configuration management
```

### WebSocket Events
- `agent:status` - Agent status updates
- `workflow:progress` - Workflow execution progress
- `system:metrics` - Real-time system metrics
- `chat:message` - Chat message streaming

## 🧪 Development

### Available Scripts

```bash
# Development
npm run dev              # Start development servers
npm run dev:client       # Start only client dev server
npm run dev:server       # Start only server dev server

# Building
npm run build            # Build client for production
npm run build:client     # Build only client
npm run preview          # Preview production build

# Production
npm start                # Start production server

# Utilities
npm run lint             # Lint code
npm run type-check       # TypeScript type checking
```

### Code Structure

```typescript
// Component example
import React from 'react'
import { useAgent } from '../contexts/AgentContext'

const AgentCard: React.FC<AgentCardProps> = ({ agent }) => {
  const { selectAgent } = useAgent()
  
  return (
    <div className="card p-6">
      {/* Component content */}
    </div>
  )
}
```

## 🐳 Docker Deployment

### Production Container
```bash
# Build production image
docker build -t agentic-frontend .

# Run container
docker run -p 3001:3001 agentic-frontend
```

### Development Container
```bash
# Start development environment
docker-compose --profile dev up
```

## 🔍 Monitoring and Debugging

### Health Checks
- **Application Health**: `/health` endpoint
- **Service Dependencies**: Backend connectivity checks
- **WebSocket Status**: Real-time connection monitoring

### Logging
- **Structured Logging**: JSON format with log levels
- **Request Logging**: API request/response tracking
- **Error Tracking**: Comprehensive error reporting

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines
- Follow TypeScript best practices
- Use React hooks and functional components
- Implement proper error handling
- Add comprehensive type definitions
- Write meaningful commit messages

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the documentation in the `docs/` directory
- Review the API documentation
- Check existing issues and discussions
- Create a new issue for bugs or feature requests

---

**Built with ❤️ for the Agentic AI ecosystem**
