from sqlalchemy import text
from ..db.db import SessionLocal
from langchain_core.documents import Document


class PgVectorRetriever:
    def __init__(self, embeddings, top_k=10):
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
