# ISOGuideBot – ISO 27001 Chatbot

A simple decision-support chatbot that answers questions about **ISO/IEC 27001 information security controls** using a **RAG (Retrieval-Augmented Generation)** pipeline.

Built with:
- **FastAPI** (backend)
- **ChromaDB** (vector database)
- **SentenceTransformers** (embeddings)
- **HTML/CSS/JavaScript** (frontend)

Fully open-source. No external LLM APIs.

---

## 📌 Features

- Ask questions about ISO 27001  
  (e.g., clean desk policy, access control)
- Retrieves relevant sections from your dataset  
- Local vector search (ChromaDB)  
- Simple `/ask` backend API  
- Clean chat-style frontend  

---

## 📁 Project Structure

ISO27001-Chatbot/
│
├── backend/
│ ├── main.py # FastAPI backend
│ ├── rag_setup.py # Vector DB builder
│
├── frontend/
│ ├── index.html # Chat UI
│
├── data/
│ ├── iso27001.txt # Dataset
│ ├── chroma_db/ # Vector store

yaml
Copy code

---

## ⚙️ Setup

### 1. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
2. Install dependencies
bash
Copy code
pip install -r requirements.txt
3. Build vector database
bash
Copy code
python backend/rag_setup.py
▶️ Run Backend
bash
Copy code
uvicorn backend.main:app --reload --port 8001
Open API docs:

arduino
Copy code
http://127.0.0.1:8001/docs
💬 Run Frontend
bash
Copy code
cd frontend
python -m http.server 5500
Open in browser:

arduino
Copy code
http://127.0.0.1:5500/index.html
🧪 Example Query
json
Copy code
{
  "question": "What is clean desk policy?",
  "top_k": 3
}

🎓 Course
Decision Support Systems – 2025
