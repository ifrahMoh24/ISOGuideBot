from pathlib import Path
from typing import List

import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ---------- Paths & Constants ----------

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "iso27001_controls"

# ---------- Load models & DB at startup ----------

print("🔹 Loading embedding model for API ...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("🔹 Connecting to ChromaDB ...")
client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_or_create_collection(name=COLLECTION_NAME)

print(f"   → Collection has {collection.count()} documents")


# ---------- FastAPI setup ----------

app = FastAPI(
    title="ISOGuideBot API",
    version="0.1.0",
    description="ISO 27001 Decision Support Chatbot backend (RAG-only version).",
)

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


class AskResponse(BaseModel):
    question: str
    answer: str
    contexts: List[str]


# ---------- Helper: retrieve relevant chunks ----------

def retrieve_context(question: str, top_k: int = 3) -> List[str]:
    """Embed the question and retrieve relevant ISO 27001 chunks from Chroma."""
    query_embedding = embedding_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    docs = results.get("documents", [[]])[0]
    return docs


# ---------- Endpoints ----------

@app.get("/")
def root():
    return {
        "message": "ISOGuideBot API is running. Use POST /ask to query ISO 27001 controls."
    }


@app.post("/ask", response_model=AskResponse)
def ask_iso_bot(payload: AskRequest):
    question = payload.question
    top_k = payload.top_k

    contexts = retrieve_context(question, top_k=top_k)

    if not contexts:
        return AskResponse(
            question=question,
            answer="I could not find any relevant ISO 27001 guidance for this question.",
            contexts=[],
        )

    best_context = contexts[0]

    return AskResponse(
        question=question,
        answer=best_context,
        contexts=contexts,
    )
