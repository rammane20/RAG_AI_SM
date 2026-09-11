# RAG_AI_SM
Autonomous AI Agents &amp; Hybrid Retrieval-Augmented Generation (RAG) system featuring Dense + BM25 search with RRF scoring, Corrective RAG (CRAG), interactive Streamlit web UI, and CLI assistant by Suyash Mane.






# 🧠 AI-SM: Advanced Intelligence & Retrieval-Augmented Generation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/Framework-LangChain-brightgreen?logo=chainlink" alt="LangChain" />
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Search-Hybrid%20RRF-orange" alt="Hybrid RRF" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License" />
  <img src="https://img.shields.io/badge/Author-Suyash%20Mane-informational" alt="Author" />
</p>

<p align="center">
  <b>A production-ready framework for building Autonomous AI Agents, Hybrid Retrieval-Augmented Generation (RAG) systems, and LLM-powered applications.</b>
</p>

---

## 👨‍💻 Author & Maintainer
- **Creator & Lead Developer:** **Suyash Mane**
- **Email:** [suyashmane97@gmail.com](mailto:suyashmane97@gmail.com)
- **Project Name:** **AI-SM** (Advanced Intelligence - Suyash Mane)
- **Version:** `3.1.0`
- **License:** [MIT License](./LICENSE)

---

## 📌 Architectural Overview

```
                          ┌───────────────────────────┐
                          │     User Natural Query    │
                          └─────────────┬─────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │   Conversational Router     │
                         │   (Greeting / Intent Check) │
                         └──────────────┬──────────────┘
                                        │
                   ┌────────────────────┴────────────────────┐
                   │                                         │
        ┌──────────▼──────────┐                   ┌──────────▼──────────┐
        │ Dense Vector Search │                   │  Sparse BM25 Search │
        │ (Semantic Meaning)  │                   │  (Exact Keywords)   │
        └──────────┬──────────┘                   └──────────┬──────────┘
                   │                                         │
                   └────────────────────┬────────────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │ Reciprocal Rank Fusion(RRF) │
                         │ RRF = 1 / (60 + Rank)       │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │    Corrective RAG (CRAG)    │
                         │    Relevance & Grounding    │
                         │  [CORRECT|AMBIGUOUS|LOW]    │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │     Grounded Generation     │
                         │    (With Source Citations)  │
                         └─────────────────────────────┘
```

---

## ✨ Key Features in AI-SM v3.1.0

### 1. 🔍 Hybrid Search Engine (`src/ai_sm/hybrid_rag.py`)
- **Dual-Strategy Retrieval**: Combines semantic dense embeddings with lexical BM25 keyword matching.
- **Reciprocal Rank Fusion (RRF)**: Normalizes and blends disparate retrieval scores to eliminate ranking bias:
  $$\text{RRF\_Score}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)} \quad (k = 60)$$
- Works locally out-of-the-box with TF-IDF / cosine similarity and provides seamless plug-and-play support for OpenAI, DeepSeek, and HuggingFace models.

### 2. 🛡️ Corrective RAG (CRAG) Pipeline (`src/ai_sm/corrective_rag.py`)
- Evaluates retrieved document relevance dynamically before response synthesis.
- Categorizes query confidence into:
  - **CORRECT**: Direct high-confidence knowledge synthesis with citations.
  - **AMBIGUOUS**: Strips irrelevant sentences to focus generation.
  - **LOW CONFIDENCE**: Flags missing evidence and provides recommended query suggestions.

### 3. 📂 Pre-Loaded Knowledge Base (`data/`)
A ready-to-query dataset spanning multiple formats:
- **`data/ai_agents_architecture.md`**: Guide on ReAct loops, planning agents, and memory buffers.
- **`data/rag_best_practices.json`**: Structured dataset on chunking strategies, embeddings, and RAG evaluation metrics.
- **`data/vector_databases_overview.csv`**: Comparison matrix of ChromaDB, Pinecone, FAISS, Qdrant, Weaviate, and Milvus.
- **`data/llm_engineering_handbook.txt`**: Production handbook on context window management and prompt caching.

### 4. 💻 Dual User Interfaces
- **Web App (Streamlit)**: Live conversational UI with real-time source inspection, similarity scores, and parameter sliders.
- **Terminal CLI**: Fast, responsive command-line assistant supporting both one-off questions (`--query`) and interactive chat.

### 5. 📚 Extensive Educational Notebooks & Snippets
- **LangChain v1.0 & LCEL**: Runnables, custom tools, LangSmith tracing, and DeepSeek integration.
- **Vectorstore Deep Dives**: Hands-on walkthroughs with ChromaDB, Pinecone, FAISS, and MongoDB.
- **Audio & Transformers**: Whisper-v3 transcription, Gradio demos, and model fine-tuning with HuggingFace Accelerate.

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/<your-username>/AI-SM.git
cd AI-SM

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

# Install dependencies:
pip install -r requirements.txt
# Or using pip with pyproject.toml:
pip install .
```

### 2. Run the Streamlit Web Application
Launch the web interface using the dedicated launcher:

```bash
python run_app.py
```
*Or directly with Streamlit:*
```bash
streamlit run src/ai_sm/streamlit_app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

### 3. Run the Interactive CLI Assistant
Ask questions directly from your terminal:

```bash
# Ask a single question:
python src/ai_sm/cli_app.py --query "What is the ReAct loop in AI agents?"

# Or enter continuous chat mode:
python src/ai_sm/cli_app.py
```

---

## 📁 Repository Structure

```text
AI-SM/
├── data/                                  # Knowledge base documents
│   ├── ai_agents_architecture.md          # Autonomous agents & memory guide
│   ├── llm_engineering_handbook.txt       # Production LLM engineering handbook
│   ├── rag_best_practices.json            # Chunking, embeddings & metrics
│   └── vector_databases_overview.csv      # Vector databases comparison table
├── docs/                                  # Extended documentation & cheatsheets
│   ├── Continued Education/               # Research papers (PDFs)
│   ├── prompt-cheatsheet.md               # Prompt engineering cheatsheet
│   └── README.md                          # Docs overview
├── run_app.py                             # Dedicated Web UI launcher
├── src/
│   ├── ai_sm/                             # Modern AI-SM Core Modules
│   │   ├── __init__.py                    # Package exports
│   │   ├── cli_app.py                     # Interactive CLI Assistant
│   │   ├── corrective_rag.py              # Corrective RAG (CRAG) pipeline
│   │   ├── hybrid_rag.py                  # Dense + BM25 Hybrid RAG engine
│   │   └── streamlit_app.py               # Streamlit web application
│   ├── langchain/                         # LangChain components & notebooks
│   │   ├── codesnippets/                  # Memory, chat, and retrieval snippets
│   │   ├── notebooks/                     # DeepSeek, RAG, and LCEL tutorials
│   │   └── packages/                      # Local PDF & directory loaders
│   ├── opai/                              # OpenAI API examples (TTS, QA)
│   └── transformers/                      # Whisper transcription & fine-tuning
├── tests/                                 # Unit and integration test suite
│   ├── test_ai_sm.py                      # AI-SM test suite (All passing)
│   └── test_query_local_docs.py           # Local doc query tests
├── LICENSE                                # MIT License (Suyash Mane)
├── pyproject.toml                         # Project metadata & dependencies
└── README.md                              # This file
```

---

## 🧪 Testing

Run the automated test suite to verify the indexing, hybrid retrieval, and CRAG modules:

```bash
python -m unittest tests/test_ai_sm.py
```

---

## 📤 Uploading to GitHub

To publish this project to your GitHub account, run the following commands in the project root:

```bash
# 1. Initialize Git repository
git init

# 2. Add all project files
git add .

# 3. Create your first commit
git commit -m "Initial commit: AI-SM by Suyash Mane"

# 4. Set main branch
git branch -M main

# 5. Connect your GitHub repository (replace with your repository URL)
git remote add origin https://github.com/<your-github-username>/AI-SM.git

# 6. Push to GitHub
git push -u origin main
```

---

## 📜 License

Copyright © 2025-2026 **Suyash Mane** ([suyashmane97@gmail.com](mailto:suyashmane97@gmail.com)).  
Distributed under the **[MIT License](./LICENSE)**.
