# Docker-based RAG Q&A System

**A professional RAG (Retrieval-Augmented Generation) pipeline integrating local embedding models with cloud-based LLMs for deep academic paper analysis.**

---

## 🚀 Overview

This project implements a robust **RAG (Retrieval-Augmented Generation)** system encapsulated in a **Docker** environment. It is specifically designed to parse, index, and query complex academic documents (such as the DINOv2 paper). 

By leveraging **local BGE embedding models** for privacy and cost-efficiency, and the **DeepSeek API** for high-quality reasoning, this system provides precise, context-aware answers grounded in provided PDF data.

## 🛠️ Tech Stack

- **Orchestration:** Docker & Docker Compose
- **Framework:** LangChain
- **LLM:** DeepSeek-V3 (via API)
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Embedding:** `bge-small-en-v1.5` (Running locally)
- **Frontend:** Streamlit
- **Environment:** WSL2 (Ubuntu 22.04)

## 🏗️ System Architecture

The system operates in a hybrid compute mode:
1. **Offline Ingestion:** PDF is parsed and split using `RecursiveCharacterTextSplitter`. Chunks are converted into vectors via a local BGE model and stored in **FAISS**.
2. **Online Retrieval:** User queries are embedded locally, and a similarity search is performed to retrieve relevant context.
3. **Generation:** The retrieved context + user query are sent to **DeepSeek LLM** to generate the final grounded response.

## 📦 Installation & Setup

### Prerequisites
- Docker & Docker Compose installed
- A valid DeepSeek API Key

### Step 1: Clone the Repository
```bash
git clone [https://github.com/djdj-student/AI_RAG_PROJECT.git](https://github.com/djdj-student/AI_RAG_PROJECT.git)
cd AI_RAG_PROJECT

Step 2: Configuration
Create a .env file in the root directory and add your API key:
DEEPSEEK_API_KEY=your_actual_api_key_here

(Note: The .env file is ignored by git for security.)
Step 3: Deployment
Launch the system using Docker Compose:
docker compose up --build

Step 4: Access the UI
Once the container is running, open your browser and navigate to:
http://localhost:7887

📸 Showcase & Results
System Performance
The system was tested against the DINOv2 research paper with three benchmarks:
 * Summarization: Accurately extracted core contributions.
 * Detail Retrieval: Successfully identified specific datasets used in experiments.
 * Logic Reasoning: Explained complex architectural optimizations regarding color bias.
📝 Lessons Learned
 * Version Control: Encountered and resolved a NameError in the transformers library by implementing strict version pinning in requirements.txt.
 * Hybrid Compute: Optimized costs by performing heavy embedding tasks locally while utilizing cloud APIs for final synthesis.
 * Engineering: Gained hands-on experience with WSL2 networking and Docker port mapping.
Author: [djdj-student]
Major: Artificial Intelligence

---
