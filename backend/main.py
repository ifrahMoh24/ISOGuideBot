"""ISOGuideBot API: semantic search over ISO/IEC 27001:2022 controls.

Runs fully locally: no external LLM API, no API key.

Run from the project root:
    uvicorn backend.main:app --reload --port 8001
"""

import os
from typing import List

import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from backend.rag_setup import COLLECTION_NAME, DB_PATH, EMBEDDING_MODEL

# ---------- Settings ----------

# Chunks further away than this (cosine distance) are treated as unrelated.
# Measured on sample queries: ISO questions land at 0.37-0.74, off-topic
# questions ("capital of France", "overtime pay") at 0.85+.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.80"))
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500"
).split(",")

NOT_COVERED = (
    "I couldn't find anything about that in the ISO 27001 guide. "
    "Try asking about a specific control, such as access control, "
    "incident management or clear desk."
)

# ---------- Load model & DB at startup ----------

print("🔹 Loading embedding model ...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("🔹 Connecting to ChromaDB ...")
client = chromadb.PersistentClient(path=str(DB_PATH))
try:
    collection = client.get_collection(name=COLLECTION_NAME)
except Exception as exc:  # the vector store was never built
    raise RuntimeError(
        "Vector store not found. Build it first with: python -m backend.rag_setup"
    ) from exc
print(f"   → Collection has {collection.count()} chunks")

# ---------- FastAPI setup ----------

app = FastAPI(
    title="ISOGuideBot API",
    version="1.0.0",
    description="ISO/IEC 27001:2022 decision-support chatbot (local semantic search).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(3, ge=1, le=10)


class Source(BaseModel):
    control: str  # e.g. "A.7.7", or "" for general sections
    title: str
    text: str
    distance: float


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]


# ---------- Retrieval ----------

def retrieve(question: str, top_k: int) -> List[Source]:
    """Embed the question and return the closest chunks within MAX_DISTANCE."""
    results = collection.query(
        query_embeddings=embedding_model.encode([question]).tolist(),
        n_results=top_k,
    )
    return [
        Source(
            control=meta.get("control", ""),
            title=meta.get("title", ""),
            text=doc,
            distance=round(dist, 3),
        )
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
        if dist <= MAX_DISTANCE
    ]


# ---------- Endpoints ----------

@app.get("/")
def root():
    return {
        "message": "ISOGuideBot API is running. POST /ask to query ISO 27001 controls.",
        "chunks": collection.count(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    question = payload.question.strip()
    sources = retrieve(question, payload.top_k)

    if not sources:
        return AskResponse(question=question, answer=NOT_COVERED, sources=[])

    # The best match is the answer; the rest are shown as related controls.
    return AskResponse(question=question, answer=sources[0].text, sources=sources)
