"""ISOGuideBot API: retrieval-augmented answers about ISO/IEC 27001:2022.

Run from the project root:
    uvicorn backend.main:app --reload --port 8001
"""

import logging
import os
from typing import List, Optional

import anthropic
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from backend.rag_setup import COLLECTION_NAME, DB_PATH, EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("isoguidebot")

# ---------- Settings ----------

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
# Chunks further away than this (cosine distance) are treated as unrelated.
# Measured on sample queries: ISO questions land at 0.37-0.74, off-topic
# questions ("capital of France", "overtime pay") at 0.85+.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.80"))
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500"
).split(",")

SYSTEM_PROMPT = """You are ISOGuideBot, an assistant that explains ISO/IEC 27001:2022 to people implementing an information security management system.

Answer using only the numbered sources in the user's message. Cite every claim with the source number in square brackets, like [1] or [2][3], and mention Annex A control numbers (for example A.7.7) when the sources give them.

If the sources do not cover the question, say that this guide does not cover it and suggest what ISO 27001 topic the user could ask about instead. Do not fill gaps from general knowledge, and do not invent control numbers.

Keep answers short: two to five sentences, or a brief list when the user asks for steps. Write plain text without Markdown headings."""

NOT_COVERED = (
    "I couldn't find anything about that in the ISO 27001 guide. "
    "Try asking about a specific control, such as access control, "
    "incident management or clear desk."
)

# ---------- Load models & DB at startup ----------

log.info("Loading embedding model %s ...", EMBEDDING_MODEL)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

log.info("Connecting to ChromaDB at %s ...", DB_PATH)
db_client = chromadb.PersistentClient(path=str(DB_PATH))
try:
    collection = db_client.get_collection(name=COLLECTION_NAME)
except Exception as exc:  # collection missing: the vector store was never built
    raise RuntimeError(
        "Vector store not found. Build it first with: python -m backend.rag_setup"
    ) from exc
log.info("Collection has %d chunks", collection.count())

# The SDK reads ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) from the environment.
# Without one, the API still works in retrieval-only mode.
llm_enabled = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
claude = anthropic.Anthropic() if llm_enabled else None
log.info("Answer mode: %s", "claude" if llm_enabled else "retrieval-only")

# ---------- FastAPI setup ----------

app = FastAPI(
    title="ISOGuideBot API",
    version="0.2.0",
    description="ISO/IEC 27001:2022 decision-support chatbot (RAG with Claude).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(4, ge=1, le=10)


class Source(BaseModel):
    id: int
    title: str
    control: str
    text: str
    distance: float


class AskResponse(BaseModel):
    question: str
    answer: str
    mode: str  # "claude", "retrieval-only" or "fallback" (Claude call failed)
    sources: List[Source]


# ---------- Retrieval ----------

def retrieve(question: str, top_k: int) -> List[Source]:
    """Embed the question and return the closest chunks that pass MAX_DISTANCE."""
    results = collection.query(
        query_embeddings=embedding_model.encode([question]).tolist(),
        n_results=top_k,
    )
    sources = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        if dist > MAX_DISTANCE:
            continue
        sources.append(
            Source(
                id=len(sources) + 1,
                title=meta.get("title", ""),
                control=meta.get("control", ""),
                text=doc,
                distance=round(dist, 3),
            )
        )
    return sources


# ---------- Generation ----------

def generate_answer(question: str, sources: List[Source]) -> str:
    """Ask Claude to answer from the retrieved sources, with [n] citations."""
    numbered = "\n\n".join(f"[{s.id}] {s.text}" for s in sources)
    response = claude.beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Sources:\n\n{numbered}\n\nQuestion: {question}",
            }
        ],
        # Short factual Q&A does not need deep reasoning.
        output_config={"effort": "low"},
        # If a safety classifier declines, the API retries on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        return "I can't answer that question."
    return "".join(b.text for b in response.content if b.type == "text").strip()


def retrieval_only_answer(sources: List[Source]) -> str:
    return sources[0].text


# ---------- Endpoints ----------

@app.get("/")
def root():
    return {
        "message": "ISOGuideBot API is running. POST /ask to query ISO 27001 controls.",
        "mode": "claude" if llm_enabled else "retrieval-only",
        "chunks": collection.count(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    question = payload.question.strip()
    sources = retrieve(question, payload.top_k)

    if not sources:
        return AskResponse(question=question, answer=NOT_COVERED, mode="none", sources=[])

    if not llm_enabled:
        return AskResponse(
            question=question,
            answer=retrieval_only_answer(sources),
            mode="retrieval-only",
            sources=sources,
        )

    mode = "claude"
    try:
        answer = generate_answer(question, sources)
    except anthropic.AuthenticationError:
        log.error("Claude API key was rejected; answering from retrieval only")
        answer, mode = retrieval_only_answer(sources), "fallback"
    except anthropic.RateLimitError:
        log.warning("Claude rate limit hit; answering from retrieval only")
        answer, mode = retrieval_only_answer(sources), "fallback"
    except anthropic.APIStatusError as exc:
        log.error("Claude API error %s: %s", exc.status_code, exc.message)
        answer, mode = retrieval_only_answer(sources), "fallback"
    except anthropic.APIConnectionError:
        log.error("Could not reach the Claude API; answering from retrieval only")
        answer, mode = retrieval_only_answer(sources), "fallback"

    return AskResponse(question=question, answer=answer, mode=mode, sources=sources)
