from fastapi import FastAPI
from .rag.agent import MedicalAgent

app = FastAPI(title="Healthcare Agentic RAG API")
agent = MedicalAgent()


@app.post("/ask")
def ask_question(query: str) -> str:
    response = agent.run(query)
    return response
