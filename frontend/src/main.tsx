/// <reference types="vite/client" />

declare global {
  interface ImportMetaEnv {
    readonly VITE_API_URL: string
    readonly VITE_WS_URL: string
    readonly VITE_SERVER_URL: string
    readonly REACT_APP_API_URL: string
    readonly REACT_APP_WS_URL: string
  }

  interface ImportMeta {
    readonly env: ImportMetaEnv
  }
}

import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './utils/logger' // Initialize the logger

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
