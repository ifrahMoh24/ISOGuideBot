# ISOGuideBot – ISO 27001 Chatbot

A simple decision-support chatbot that answers questions about **ISO/IEC 27001:2022 information security controls** using semantic search over a local vector database (the retrieval part of a **RAG** pipeline).

Built with:
- **FastAPI** (backend)
- **ChromaDB** (vector database)
- **SentenceTransformers** `all-MiniLM-L6-v2` (embeddings)
- **HTML/CSS/JavaScript** (frontend)

Fully open-source and runs locally. No external LLM APIs, no API key.

---

## 📌 Features

- Ask questions about ISO 27001 (e.g. clean desk policy, MFA, incident response)
- Covers all **93 Annex A controls** of ISO 27001:2022 in 4 themes, plus ISMS clauses 4–10, the Statement of Applicability and the changes from the 2013 version
- Returns the best-matching control and lists related controls (e.g. `A.8.5`)
- Tells you when a question is outside the guide instead of returning an unrelated answer
- Simple `/ask` backend API and a clean chat-style frontend

---

## ⚙️ How it works

1. **Build once** (`rag_setup.py`): the dataset is split into one chunk per control, each chunk is embedded with MiniLM, and stored in ChromaDB with its control number as metadata.
2. **On each question** (`main.py`): the question is embedded, the nearest chunks are found by cosine distance, and chunks further than `0.80` are dropped.
3. **Answer**: the closest control is returned as the answer and the others as related controls. If nothing is close enough, the bot says the topic isn't covered.

The `0.80` cutoff was chosen by measuring sample queries: ISO questions scored 0.37–0.74, off-topic ones ("What is the capital of France?") scored 0.85 or more.

---

## 📁 Project Structure

```
ISO27001-Chatbot/
├── backend/
│   ├── main.py          # FastAPI backend
│   └── rag_setup.py     # Vector DB builder
├── frontend/
│   └── index.html       # Chat UI
├── data/
│   ├── iso27001.txt     # Dataset (ISO 27001:2022 Annex A)
│   └── chroma_db/       # Generated vector store (git-ignored)
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Build the vector database

```bash
python -m backend.rag_setup
```

Re-run this whenever `data/iso27001.txt` changes.

---

## ▶️ Run the backend

```bash
uvicorn backend.main:app --reload --port 8001
```

API docs: <http://127.0.0.1:8001/docs>

## 💬 Run the frontend

In a second terminal:

```bash
cd frontend
python3 -m http.server 5500
```

Open <http://127.0.0.1:5500/index.html>

If port 8001 is busy, run the backend on another port and open `index.html?api=http://127.0.0.1:8002`.

---

## 🧪 Example query

`POST /ask`

```json
{
  "question": "Do we need MFA?",
  "top_k": 3
}
```

Response (shortened):

```json
{
  "question": "Do we need MFA?",
  "answer": "A.8.5 Secure authentication (Technological controls)\nSecure authentication technologies and procedures must be implemented ...",
  "sources": [
    { "control": "A.8.5", "title": "A.8.5 Secure authentication (Technological controls)", "text": "...", "distance": 0.492 }
  ]
}
```

---

## 📝 Notes

- The dataset descriptions are plain-language summaries written for this project, not the official text of the standard.
- Possible next steps: generate answers with an LLM from the retrieved controls, and build a small evaluation set to measure retrieval accuracy.

---

## 🎓 Course

Decision Support Systems – 2025
