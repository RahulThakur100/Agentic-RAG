import React from 'react'
import ChatBox from './components/ChatBox'
import './App.css'

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>🏥 Agentic RAG - Medical Guidelines</h1>
        <p>Ask questions about WHO medical guidelines</p>
      </header>
      <ChatBox />
    </div>
  )
}

export default App

