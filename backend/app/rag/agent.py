from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .retriever import PgVectorRetriever


class MedicalAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.retriever = PgVectorRetriever(embeddings=self.embeddings)

    def run(self, query: str) -> str:
        """
        Agentic Rag flow:
        1. Retrieve top douments from pgvector
        2. Feed into an LLM with structured prompt
        3. Return the answer
        """

        docs = self.retriever.get_relevant_information(query)
        print(f"Retrieved {len(docs)} documents for the query.")

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""You are a medical guideline assistant.Answer the question
                    using ONLY the information in the context below.

                    If the context does not contain the answer, say:
                    "I don't have enough information to answer that."

                    QUESTION:
                    {query}

                    CONTEXT:
                    {context}

                    ANSWER:
                """
        response = self.llm.invoke(prompt)

        print("[RAG] LLM Response:\n", response)
        return response.content
