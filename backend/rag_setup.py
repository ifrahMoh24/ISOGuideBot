import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_PATH = BASE_DIR / "data" / "iso27001.txt"
DB_PATH = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "iso27001_controls"


def load_document() -> str:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"iso27001.txt not found at {DATA_PATH}")
    return DATA_PATH.read_text(encoding="utf-8")


def chunk_text(text: str, max_chars: int = 600) -> list[str]:
    """
    Very simple chunker: splits on double newlines, then further splits
    long blocks into ~max_chars chunks.
    """
    raw_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    chunks: list[str] = []

    for block in raw_blocks:
        if len(block) <= max_chars:
            chunks.append(block)
        else:
            # Hard split into fixed-size pieces
            start = 0
            while start < len(block):
                end = start + max_chars
                chunks.append(block[start:end])
                start = end

    return chunks


def build_vector_store():
    print(f"🔹 Loading document from {DATA_PATH} ...")
    text = load_document()

    print("🔹 Chunking document ...")
    chunks = chunk_text(text)
    print(f"   → {len(chunks)} chunks created")

    print("🔹 Loading embedding model ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("🔹 Creating ChromaDB client ...")
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))

    # We will manage embeddings manually (no custom embedding_function)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Clear old data if re-running
    if collection.count() > 0:
        print("   → Collection already has data, deleting existing entries...")
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print("🔹 Embedding and inserting chunks ...")
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    embeddings = model.encode(chunks).tolist()

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": "iso27001"} for _ in chunks],
    )

    print("✅ Done. Vector store created at:", DB_PATH)
    print("   Collection name:", COLLECTION_NAME)
    print("   Total documents in collection:", collection.count())


def quick_test(query: str = "What is clean desk policy?"):
    """
    Simple retrieval test to check that our DB works.
    """
    print("🔹 Running quick retrieval test ...")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        print("⚠️ Collection is empty. Run build_vector_store() first.")
        return

    print(f"   Query: {query}")
    # For this simple test, we let Chroma use its default embedding; not ideal,
    # but good enough to see if something comes back.
    results = collection.query(
        query_texts=[query],
        n_results=3,
    )

    docs = results.get("documents", [[]])[0]
    print("   Top results:")
    for i, d in enumerate(docs, start=1):
        print(f"\n--- Result {i} ---")
        print(d)


if __name__ == "__main__":
    build_vector_store()
    # Optional: run a quick test
    quick_test()
