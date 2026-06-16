# Agentic RAG Pipeline

A production-ready, containerized Retrieval-Augmented Generation (RAG) API built to orchestrate intelligent search across structured and unstructured project data. 

This architecture implements a multi-tool ReAct agent capable of cross-referencing CSV supply chain trackers with raw meeting notes to identify and analyze complex project dependencies.

## 🏗️ Architecture & Tech Stack

This pipeline was engineered with a focus on speed, stability, and hardware-agnostic deployment:

* **Orchestration:** [LangGraph](https://python.langchain.com/docs/langgraph) & LangChain
* **LLM Inference:** Llama-3.3-70B-Versatile via [Groq](https://groq.com/) (Chosen for ultra-low latency inference).
* **Embedding Engine:** [FastEmbed](https://qdrant.github.io/fastembed/) (BAAI/bge-small-en-v1.5). 
* **Vector Database:** [Qdrant](https://qdrant.tech/) (Local file-based deployment for isolated environments).
* **API Layer:** FastAPI & Uvicorn.
* **Deployment:** Docker.

### 💡 Key Engineering Decisions
1. **PyTorch-Free Embeddings:** Traditional HuggingFace embeddings rely heavily on PyTorch, which is prone to OpenMP threading deadlocks on Windows/local environments. This project utilizes `FastEmbed` backed by the ONNX runtime, ensuring blazing-fast, crash-free vectorization across any OS.
2. **Stateless Agentic Loop:** The LangGraph ReAct agent is dynamically injected with a Qdrant retrieval tool. It autonomously determines when to query the database, ensuring it only fetches context when necessary rather than relying on a rigid, hardcoded semantic search pipeline.
3. **Cloud-Native Containerization:** The entire stack, including the local vector database, is packaged into a lightweight `python:3.12-slim` Docker image, completely removing "it works on my machine" dependency bottlenecks.

## 🚀 Quick Start

### Option 1: Run via Docker (Recommended)
You can spin up the entire API in an isolated container without installing Python dependencies.

1. Clone the repository and navigate to the root directory.
2. Build the image:
   ```bash
   docker build -t nexusolve-rag .
