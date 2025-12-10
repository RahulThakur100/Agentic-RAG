from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .rag.agent import MedicalAgent

app = FastAPI(title="Healthcare Agentic RAG API")
agent = MedicalAgent()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Vite default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask")
def ask_question(query: str) -> str:
    response = agent.run(query)
    return response
