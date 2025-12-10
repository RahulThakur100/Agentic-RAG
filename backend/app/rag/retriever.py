from sqlalchemy import text
from ..db.db import SessionLocal
from langchain_core.documents import Document
from langchain_core.tools import tool


class PgVectorRetriever:
    def __init__(self, embeddings, top_k=5):
        self.embeddings = embeddings
        self.top_k = top_k

    def get_relevant_information(self, query: str):
        """Embeds a query and retrieves top similar documents using pgvector."""
        query_embedding = self.embeddings.embed_query(query)

        # Pass embedding as list - psycopg2 will convert it to PostgreSQL array format
        # pgvector accepts embeddings as arrays, and SQLAlchemy/psycopg2 handles the conversion
        sql = text("""
            SELECT id, file_name, chunk, embedding <=> (:query_embedding)::vector AS distance
            FROM test.documents
            ORDER BY embedding <=> (:query_embedding)::vector
            LIMIT :top_k
        """)

        with SessionLocal() as session:
            rows = session.execute(
                sql,
                {"query_embedding": query_embedding, "top_k": self.top_k}
            ).fetchall()

        docs = []
        for row in rows:
            docs.append(
                Document(
                    page_content=row.chunk,
                    metadata={"id": row.id, "file_name": row.file_name,
                              "distance": row.distance}
                )
            )
        return docs


def create_retrieval_tool(retriever: PgVectorRetriever):
    """Create a LangChain tool from the retriever for agent use."""

    @tool
    def search_medical_documents(query: str) -> str:
        """
        Search medical documents and guidelines from WHO documents stored in the database.

        Use this tool when you need to find information about:
        - Medical conditions, symptoms, or diseases
        - Treatment guidelines and protocols
        - Diagnostic procedures
        - Medication information
        - Medical best practices

        Args:
            query: The search query describing what medical information you need to find.

        Returns:
            A string containing the relevant document chunks found, separated by newlines.
            Each chunk includes the content and metadata about which document it came from.
        """
        docs = retriever.get_relevant_information(query)

        if not docs:
            return "No relevant documents found for this query."

        # Format the results for the agent
        results = []
        for i, doc in enumerate(docs, 1):
            results.append(
                f"[Document {i} - Source: {doc.metadata.get('file_name', 'Unknown')}]\n"
                f"{doc.page_content}\n"
                f"(Similarity distance: {doc.metadata.get('distance', 'N/A'):.4f})\n"
            )

        return "\n---\n".join(results)

    return search_medical_documents
