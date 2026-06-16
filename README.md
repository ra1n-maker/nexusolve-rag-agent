# Nexusolve Agentic RAG Pipeline

A production-ready, containerized Retrieval-Augmented Generation (RAG) API built to orchestrate intelligent search across structured and unstructured project data. 

This architecture implements a multi-tool ReAct agent capable of cross-referencing CSV supply chain trackers with raw meeting notes to identify and analyze complex project dependencies.

## 🏗️ Architecture & Tech Stack

This pipeline was engineered with a focus on speed, stability, and hardware-agnostic deployment:

* **Orchestration:** [LangGraph](https://python.langchain.com/docs/langgraph) & LangChain
* **LLM Inference:** Llama-3.3-70B-Versatile via [Groq](https://groq.com/) (Chosen for ultra-low latency inference and zero local VRAM dependency).
* **Embedding Engine:** [FastEmbed](https://qdrant.github.io/fastembed/) (BAAI/bge-small-en-v1.5). 
* **Vector Database:** [Qdrant](https://qdrant.tech/) (Local file-based deployment for isolated environments).
* **API Layer:** FastAPI & Uvicorn.
* **Deployment:** Docker.

### 💡 Key Engineering Decisions
1. **PyTorch-Free Embeddings:** Traditional embedding pipelines often rely heavily on PyTorch, which is prone to OpenMP threading deadlocks on local Windows environments. This project explicitly utilizes `FastEmbed` backed by the ONNX runtime, ensuring blazing-fast, crash-free vectorization across any OS.
2. **Stateless Agentic Loop:** The LangGraph ReAct agent is dynamically injected with a Qdrant retrieval tool. It autonomously determines when to query the database, ensuring it only fetches context when necessary rather than relying on a rigid, hardcoded semantic search pipeline.
3. **Cloud-Native Containerization:** The entire stack, including the local vector database, is packaged into a lightweight `python:3.12-slim` Docker image, completely removing "it works on my machine" dependency bottlenecks.

## 🚀 Quick Start

### Option 1: Run via Docker (Recommended)
Spin up the entire API in an isolated container without installing any local Python dependencies.

1. Clone the repository and navigate to the root directory.
2. Build the image:
   \```bash
   docker build -t nexusolve-rag .
   \```
3. Run the container:
   \```bash
   # Pass your Groq API key at runtime
   docker run -d -p 8000:8000 -e GROQ_API_KEY="your_api_key_here" nexusolve-rag
   \```

### Option 2: Local Python Environment
1. Create and activate a virtual environment (`python -m venv venv`).
2. Install dependencies:
   \```bash
   pip install -r requirements.txt
   \```
3. Set your Groq API key:
   \```bash
   export GROQ_API_KEY="your_api_key_here"
   \```
4. Build the Qdrant index:
   \```bash
   python ingest.py
   \```
5. Start the API server:
   \```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   \```

## 📡 API Usage

**Endpoint:** `POST /ask`

\```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What procurement risks are affecting Project Atlas regarding Supplier X?"}'
\```

**Example Response:**
\```json
{
  "answer": "According to the project knowledge base, the bearing shipment from Supplier X has been delayed by three weeks due to customs issues (meeting_notes_apr12.txt). This is categorized as a High Risk dependency (Row 46)."
}
\```
