import os
from pypdf import PdfReader
from openai import OpenAI

from backend.app.db.db import SessionLocal
from backend.app.db.models import Document
from ..config import OPENAI_API_KEY
from pathlib import Path
import shutil

ROOT_DIR = Path(__file__).resolve().parents[3]  # go up to repository root
PDF_DIR = ROOT_DIR / "backend" / "data" / "raw_pdfs"
PROCESSED_DIR = PDF_DIR / "processed"

CHUNK_SIZE = 500  # Number of words per chunk

client = OpenAI(api_key=OPENAI_API_KEY)


def chunk_text(text: str, size: 500):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i + size])


def extract_pdf_text(file_path: str):
    reader = PdfReader(file_path)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n".join(pages)


def embed_text(text: str):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def ingest_pdfs():
    db = SessionLocal()

    for file in os.listdir(PDF_DIR):
        if file.endswith(".pdf"):
            file_path = os.path.join(PDF_DIR, file)
            print(f"Processing {file}")

            text = extract_pdf_text(file_path)

            for chunk in chunk_text(text, CHUNK_SIZE):
                embedding = embed_text(chunk)

                doc = Document(
                    file_name=file,
                    chunk=chunk,
                    embedding=embedding
                )
                db.add(doc)

            db.commit()
            print(f"Finished ingesting {file}")

            # Move the PDF to processed folder
            dest_path = PROCESSED_DIR / file
            shutil.move(str(file_path), str(dest_path))
            print(f"Moved {file} ➜ {dest_path}")

    db.close()


if __name__ == "__main__":
    ingest_pdfs()
