from sqlalchemy import text
from openai import OpenAI
from ..db.db import SessionLocal
from ..config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def get_embedding(query: str):
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    return response['data'][0].embedding


def search(query: str, top_k: int = 5):
    embedding = get_embedding(query)
    db = SessionLocal()

    sql = text("""
            SELECT chunk,
                embedding <=> (:query_embedding)::vector AS distance
            FROM test.documents
            ORDER BY embedding <=> (:query_embedding)::vector
            LIMIT :top_k
        """)

    rows = db.execute(sql, {
        "query_embedding": embedding,
        "top_k": top_k
    }).fetchall()

    db.close()

    return [row[0] for row in rows]
