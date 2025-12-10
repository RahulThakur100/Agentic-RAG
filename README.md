# Agentic RAG for Medical Guidelines

An intelligent Retrieval-Augmented Generation (RAG) system that uses **agentic AI** to autonomously reason about medical queries, decompose complex questions and iteratively retrieve information from WHO medical documents stored in a PostgreSQL vector database.

## 🎯 What Makes This Agentic?

Unlike traditional RAG systems that perform a single retrieval step, this system implements **true agentic behavior**:

1. **Autonomous Reasoning**: The agent analyzes queries and decides what actions to take
2. **Query Decomposition**: Breaks complex questions into sub-queries automatically
3. **Iterative Retrieval**: Performs multiple searches with refined queries based on results
4. **Self-Correction**: Evaluates search results and adapts strategy if needed
5. **Tool Orchestration**: Dynamically uses retrieval tools based on reasoning

### Agentic Process Flow

```
User Query
    ↓
Agent Reasoning (Step 1)
    ├─ Analyzes: "What information is needed?"
    ├─ Decides: "Should I search? What terms?"
    └─ Plans: "Break into sub-queries?"
    ↓
Tool Execution (Step 2)
    ├─ Search 1: "pneumonia symptoms"
    ├─ Evaluate results
    ├─ Search 2: "pneumonia treatment" (if needed)
    └─ Evaluate results
    ↓
Agent Reasoning (Step 3)
    ├─ "Are results sufficient?"
    ├─ "Need more specific search?"
    └─ "Can I synthesize an answer?"
    ↓
Final Answer Synthesis
```

The agent can perform up to **10 reasoning steps**, making multiple tool calls and refining its approach autonomously.

## 🏗️ Architecture

- **LLM**: GPT-4o-mini (OpenAI) for reasoning and synthesis
- **Embeddings**: text-embedding-3-small (OpenAI)
- **Vector Database**: PostgreSQL with pgvector extension
- **Agent Framework**: LangChain + LangGraph (ReAct pattern)
- **API**: FastAPI REST API

## 📋 Prerequisites

- Python 3.8+
- Node.js 18+ and npm (for frontend)
- PostgreSQL 12+ with pgvector extension
- OpenAI API key
- PDF documents to ingest (WHO guidelines)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Agentic-RAG
```

### 2. Set Up Environment

Create a `.env` file in the `backend` directory:

```env
OPENAI_API_KEY=your_openai_api_key
DB_CONN_STRING=postgresql://user:password@localhost:5432/dbname
```

### 3. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Set Up Database

```sql
-- Create database and schema
CREATE DATABASE your_db_name;
CREATE EXTENSION vector;
CREATE SCHEMA test;
```

### 5. Ingest Documents

Place PDF files in `backend/data/raw_pdfs/` and run:

```bash
python -m app.ingestion.ingest_pdfs
```

### 6. Start the API Server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### 7. Start the Frontend (Optional)

```bash
cd ../frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000` and connects to the backend API automatically.

## 📡 API Usage

### Ask a Question

```bash
curl -X POST "http://localhost:8000/ask?query=What are the symptoms and treatment for pneumonia?"
```

### Example Queries

- Simple: `"What is pneumonia?"`
- Complex: `"Compare the symptoms, diagnosis and treatment options for pneumonia and malaria"`
- Multi-part: `"What are the WHO guidelines for pneumonia management, including diagnosis and treatment?"`

## 🔍 How It Works

1. **Document Ingestion**: PDFs are chunked, embedded and stored in pgvector
2. **Query Processing**: Agent receives query and reasons about information needs
3. **Autonomous Search**: Agent decides search strategy and executes multiple retrievals
4. **Synthesis**: Agent combines information from multiple sources into final answer

## 📁 Project Structure

```
Agentic-RAG/
├── backend/              # Python FastAPI backend
│   ├── app/
│   │   ├── main.py      # FastAPI application
│   │   ├── db/          # Database models and connection
│   │   ├── ingestion/   # PDF ingestion pipeline
│   │   └── rag/         # Agentic RAG implementation
│   ├── data/raw_pdfs/   # Place PDFs here for ingestion
│   └── requirements.txt
└── frontend/            # React frontend
    ├── src/components/   # React components
    └── package.json
```

## 🛠️ Key Technologies

- **Backend**: FastAPI, LangChain, LangGraph, pgvector, PostgreSQL
- **Frontend**: React, Vite, React Markdown
- **AI**: OpenAI GPT-4o-mini, text-embedding-3-small

## 📝 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines if needed]
