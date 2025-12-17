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

# Optional: where to log MLflow runs
# Local sqlite file at project root (recommended for local dev):
MLFLOW_TRACKING_URI="sqlite:///C:/Users/YourUser/Agentic-RAG/mlflow.db"
# Or point to a remote MLflow server:
# MLFLOW_TRACKING_URI="http://localhost:5000"
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

## 📈 MLflow Tracking & Evaluation

The backend logs every inference and offline evaluation run to MLflow, so you can inspect behavior, performance, and cost over time.

### MLflow Configuration

- Tracking URI is read from `MLFLOW_TRACKING_URI` in `backend/.env` via `app.config` and `app.mlflow_logger`.
- If `MLFLOW_TRACKING_URI` is not set, the backend falls back to a local SQLite file at the project root: `mlflow.db`.

To view runs:

```bash
cd backend/..  # project root
mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db
```

Or set `MLFLOW_TRACKING_URI` and run `mlflow ui`.

### Inference Metrics (rag_inference)

Each question answered by the agent logs a run to the `rag_inference` experiment with:

- Parameters:
  - `model` (e.g. `gpt-4o-mini`)
  - `temperature`
  - `top_k` (retrieval depth)
  - `query` (user question)
  - `prompt_version` (e.g. `system_v1`)
- Metrics:
  - `latency` – end-to-end response time (seconds)
  - `retrieval_count` – how many retrieval/tool calls were made
  - `avg_chunk_distance` – mean similarity distance of retrieved chunks
  - `answer_length_tokens` – rough word-count proxy for answer length
  - `input_tokens` – prompt tokens (from usage metadata or estimated)
  - `output_tokens` – completion tokens (from usage metadata or estimated)
  - `estimated_cost_usd` – approximate OpenAI cost for the call

This gives you visibility into performance, retrieval behavior, verbosity, and cost per query.

### Retrieval Evaluation (hit_rate@k)

There is a small, fixed evaluation set under `backend/app/evaluation/`:

- `eval_dataset.json` – list of objects with:
  - `question`
  - `expected_doc_filename` (the document that should appear in the retrieved chunks)
- `run_eval.py` – runs all questions against the retriever and checks whether the expected document appears in the top‑`k` results.

Run the evaluation from the `backend` directory:

```bash
cd backend
python -m app.evaluation.run_eval
```

This logs a `rag_retrieval_eval` experiment run with:

- Params:
  - `top_k`
  - `num_questions`
- Metrics:
  - `hit_rate_at_k` – fraction of questions where the expected document was retrieved.

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
