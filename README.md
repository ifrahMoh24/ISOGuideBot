# ISOGuideBot – ISO/IEC 27001:2022 Chatbot

A decision-support chatbot that answers questions about **ISO/IEC 27001:2022 information security controls**. It uses **Retrieval-Augmented Generation (RAG)**: it first finds the relevant Annex A controls in a local vector database, then has Claude write a short answer **only from those sources, with numbered citations**.

Built with **FastAPI**, **ChromaDB**, **SentenceTransformers** (`all-MiniLM-L6-v2`), the **Claude API** and a plain **HTML/CSS/JavaScript** frontend.

## How it works

```
                 ┌──────────────── build once: python -m backend.rag_setup ───────────────┐
 data/iso27001.txt ──► chunk per control ──► embed (MiniLM) ──► ChromaDB (cosine index)
                 └────────────────────────────────────────────────────────────────────────┘

 Question ──► embed ──► top-k nearest chunks ──► drop chunks with distance > 0.80
                                                        │
                       ┌────────────────────────────────┴───────────────┐
                no chunks left                                   numbered sources
                       │                                                │
          "not covered by this guide"            Claude answers from sources only, cites [n]
                                                                        │
                                              answer + sources (control IDs) ──► chat UI
```

- **Dataset**: all 93 Annex A controls of ISO 27001:2022 in the four themes (Organizational A.5, People A.6, Physical A.7, Technological A.8), plus the ISMS clauses 4–10, the Statement of Applicability and the changes from the 2013 version. The descriptions are plain-language summaries written for this project, not the official text of the standard.
- **Chunking**: one chunk per control, so each retrieved chunk carries its control number (e.g. `A.7.7`) as metadata. Longer sections are split on sentence boundaries with the title repeated.
- **Grounding**: the system prompt tells Claude to answer only from the retrieved sources, cite each claim, and say when the guide doesn't cover a question instead of guessing.
- **Relevance cutoff**: measured on sample queries, ISO questions land at cosine distance 0.37–0.74 and off-topic ones ("capital of France") at 0.85+, so chunks beyond 0.80 are discarded and off-topic questions never reach the LLM.
- **Graceful degradation**: with no API key, or if the Claude call fails, the API still answers with the best-matching section (`mode: "retrieval-only"` / `"fallback"`).

## Project structure

```
ISO27001-Chatbot/
├── backend/
│   ├── main.py         # FastAPI app: retrieval + Claude generation
│   └── rag_setup.py    # Builds the ChromaDB vector store
├── frontend/
│   └── index.html      # Chat UI with cited sources
├── data/
│   ├── iso27001.txt    # Dataset (ISO 27001:2022 Annex A)
│   └── chroma_db/      # Generated vector store (git-ignored)
└── requirements.txt
```

## Setup

Requires Python 3.9+.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Build the vector store (re-run whenever `data/iso27001.txt` changes):

```bash
python -m backend.rag_setup
```

Set your Claude API key (optional: without it the bot runs in retrieval-only mode):

```bash
export ANTHROPIC_API_KEY=your-key-here
```

## Run

Backend (from the project root):

```bash
uvicorn backend.main:app --reload --port 8001
```

Interactive API docs: <http://127.0.0.1:8001/docs>

Frontend, in a second terminal:

```bash
cd frontend
python3 -m http.server 5500
```

Then open <http://127.0.0.1:5500/index.html>.

## API

`POST /ask`

```json
{ "question": "Do we need MFA?", "top_k": 4 }
```

Response:

```json
{
  "question": "Do we need MFA?",
  "answer": "ISO 27001:2022 expects secure authentication based on your access control policy, and MFA is the usual measure for critical systems and remote access [1]. ...",
  "mode": "claude",
  "sources": [
    { "id": 1, "control": "A.8.5", "title": "A.8.5 Secure authentication (Technological controls)", "text": "...", "distance": 0.492 }
  ]
}
```

`mode` is `claude`, `retrieval-only` (no API key), `fallback` (Claude call failed) or `none` (nothing relevant found).

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | Enables Claude answers |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Model used for generation |
| `MAX_DISTANCE` | `0.80` | Cosine distance cutoff for relevant chunks |
| `ALLOWED_ORIGINS` | `http://127.0.0.1:5500,http://localhost:5500` | CORS origins allowed to call the API |

## Limitations and next steps

- The dataset is a summary of 93 controls, not the full standard or ISO 27002 implementation guidance.
- No retrieval evaluation set yet: the next step is a small set of question → expected-control pairs to measure hit rate and tune `top_k` and the cutoff.
- Single-turn only: each question is answered independently, without chat history.
- For production: authentication and rate limiting on `/ask`, a hosted vector store, and streaming responses.

## Course

Decision Support Systems – 2025
