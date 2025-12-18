# 🧠 DocuMind

### Intelligent Document Q&A System with Hierarchical RAG, Fact Verification & Self-Learning Capabilities

> **Transform any document collection into an intelligent, context-aware knowledge base that learns and improves over time.**

DocuMind is a production-ready Retrieval-Augmented Generation (RAG) system that combines hierarchical document processing, neural re-ranking, persistent conversational memory, automated fact verification, and research paper integration. Built for both privacy-conscious local deployment and cloud-enhanced self-learning scenarios.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.14+-orange.svg)](https://github.com/run-llama/llama_index)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-purple.svg)](https://www.trychroma.com/)
[![LM Studio](https://img.shields.io/badge/LM%20Studio-Local%20LLM-green.svg)](https://lmstudio.ai/)
[![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## 🎯 Key Highlights

| Feature | Description |
|---------|-------------|
| 🏛️ **Hierarchical RAG Pipeline** | Multi-level document chunking (2048→512→128 tokens) with auto-merging retrieval |
| 💬 **Persistent Conversational Memory** | Sessions survive restarts with intelligent summarization |
| 🔍 **Gemini Fact Verification** | Periodic cross-validation to detect conflicts and outdated information |
| 📚 **Arxiv Research Integration** | Automatic paper fetching to supplement answers with latest research |
| 💾 **Semantic Query Caching** | Embedding-based similarity matching for instant responses |
| 📊 **Built-in Analytics Dashboard** | Track query patterns, source usage, and response quality |
| 🔒 **Dual Operation Modes** | Privacy-focused local mode vs feature-rich self-learning mode |

---

## ✨ Features

### 🔄 Intelligent Document Processing
- **Hierarchical Chunking**: Three-tier parsing preserves document structure and context hierarchy
- **Auto-Merging Retrieval**: Dynamically combines related chunks when accessed together
- **Neural Re-ranking**: Cross-encoder ensures only the most relevant passages reach the LLM
- **GPU Acceleration**: CUDA-optimized embeddings and reranking for sub-second processing

### 💬 Conversational Intelligence
- **Persistent Chat History**: All conversations saved to disk — resume any session
- **Automatic Summarization**: Older messages condensed into rolling summaries
- **Context-Aware Responses**: Follow-up questions understand previous conversation
- **Session Management**: List, resume, export, or clear sessions via commands

### 🧠 Self-Learning & Verification
- **Gemini Fact Verification**: Every 5 queries, cross-validates answers against Gemini
- **Conflict Detection**: Identifies contradictions between documents and current knowledge
- **Dynamic Database Updates**: Apply corrections and new knowledge directly to vector store
- **Arxiv Integration**: Fetches relevant research papers and can add them to knowledge base

### 📊 Performance & Analytics
- **Semantic Query Cache**: Similar questions (>92% similarity) return cached responses
- **Gemini Response Cache**: Verification results cached to reduce API calls
- **Query Analytics Dashboard**: Track usage patterns, cache hit rates, and ratings
- **Feedback Collection**: Rate responses 1-5 stars to track quality over time

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Document Ingestion                               │
│    PDF/Text → Hierarchical Parser (2048→512→128) → ChromaDB + HNSW      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Query Processing Pipeline                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  Semantic   │─▶│ Auto-Merge  │─▶│   Neural    │─▶│   LM Studio     │ │
│  │   Cache     │  │  Retriever  │  │  Reranker   │  │  (Qwen3-4B)     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Enhancement Layer                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │    Chat     │  │   Gemini    │  │   Arxiv     │  │    Analytics    │ │
│  │   History   │  │  Verifier   │  │   Fetcher   │  │    Dashboard    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔐 Operation Modes

### 🔒 Local Mode
*Privacy-first, fully offline operation*
- All processing happens locally on your machine
- No data transmitted to external services
- Chat history and caching still fully functional
- Ideal for sensitive or confidential documents

### 🧠 Self-Learning Mode
*Enhanced intelligence with cloud capabilities*
- Gemini-powered fact verification every 5 queries
- Automatic Arxiv paper recommendations and integration
- Dynamic knowledge base updates from verified sources
- Continuous accuracy improvement over time

Switch between modes instantly with the `/mode` command.

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **RAG Framework** | LlamaIndex 0.14+ | Orchestration, retrieval, and query processing |
| **Vector Database** | ChromaDB | Persistent embedding storage with HNSW indexing |
| **Local LLM** | LM Studio + Qwen3-4B-Thinking | Local inference with reasoning capabilities |
| **Embeddings** | all-mpnet-base-v2 | Semantic text encoding (768 dimensions) |
| **Reranker** | MS-MARCO-MiniLM-L-6-v2 | Cross-encoder relevance scoring |
| **Fact Verification** | Google Gemini 1.5 Flash | Cloud-based knowledge validation |
| **Research Papers** | Arxiv API | Academic paper retrieval (no API key needed) |
| **GPU Acceleration** | CUDA 12.x | Parallel embedding and reranking |

---

## 📋 Requirements

### Tested Hardware Configuration
| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA RTX 4050 Laptop (6GB VRAM) |
| **CPU** | Intel Core i5-13500H |
| **RAM** | 16GB DDR5 |
| **OS** | Windows 11 |

### Minimum Requirements
- Python 3.10+
- CUDA-compatible GPU (6GB+ VRAM recommended)
- 16GB+ RAM
- LM Studio with any 4B+ parameter model
- (Optional) Google Gemini API key for fact verification

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/iDheer/DocuMind.git
cd DocuMind
python -m venv environment

# Windows
.\environment\Scripts\activate
# Linux/Mac
source environment/bin/activate

pip install -r requirements.txt
```

### 2. Install LM Studio & Load Model

1. Download LM Studio from https://lmstudio.ai/
2. Search for **`qwen/qwen3-4b-thinking-2507`** (or any 4B+ model)
3. Download the model (GGUF format, ~3GB)
4. Go to **Local Server** tab → Start Server on port 1234

### 3. (Optional) Configure Gemini API

Create a `.env` file for fact verification:
```env
GEMINI_API_KEY=your_api_key_here
```
Get a free API key at: https://aistudio.google.com/apikey

### 4. Add Documents & Build Database

```bash
# Place PDF/text documents in data_large/ folder
python 1_build_database_advanced.py
```

### 5. Start Querying

```bash
python query_enhanced.py
```

---

## 💬 Command Reference

| Command | Description |
|---------|-------------|
| `/help` | Display all available commands |
| `/mode` | Switch between Local and Self-Learning modes |
| `/history` | View current session's conversation history |
| `/clear` | Clear chat history and start fresh |
| `/sessions` | List all saved sessions with timestamps |
| `/stats` | Display system statistics and feature status |
| `/cache` | View cache statistics or clear cache |
| `/verify` | Manually trigger Gemini fact verification |
| `/arxiv` | Toggle automatic Arxiv paper fetching |
| `/analytics` | View detailed usage analytics dashboard |
| `/feedback` | Toggle post-response feedback collection |
| `/rate` | Rate the last response (1-5 stars) |
| `/export` | Export current session to JSON file |
| `exit` | Save session and exit |

---

## 📁 Project Structure

```
DocuMind/
├── query_enhanced.py            # 🚀 Main application entry point
├── 1_build_database_advanced.py # 📦 Document processing & indexing
├── 2_query_system_advanced.py   # 🔍 Basic query interface (standalone)
├── 3_inspect_hierarchy.py       # 🔬 Database inspection utility
│
├── config.py                    # ⚙️ Centralized configuration
├── chat_history.py              # 💬 Conversation persistence & summarization
├── gemini_verifier.py           # ✅ Fact verification engine
├── arxiv_fetcher.py             # 📚 Research paper integration
├── db_updater.py                # 🔄 Dynamic database modification
├── cache_manager.py             # 💾 Multi-level caching system
├── analytics.py                 # 📊 Usage tracking & metrics
│
├── requirements.txt             # 📋 Python dependencies
├── .env.example                 # 🔑 API key template
├── data_large/                  # 📂 Input documents (user-provided)
├── chroma_db_advanced/          # 🗄️ Vector database (auto-generated)
├── chat_history/                # 💬 Saved sessions (auto-generated)
└── arxiv_cache/                 # 📚 Cached papers (auto-generated)
```

---

## 🎯 Example Session

```
============================================================
💬 Ready to Query! 🧠 SELF-LEARNING MODE
============================================================

❓ Question: What is the difference between a process and a thread?

🤖 Response: A process is an independent execution unit with its own 
memory space, while a thread is a lightweight execution unit within 
a process that shares memory with other threads. The OS schedules 
processes independently, but threads within the same process share 
resources like file handles and heap memory...

📄 Sources:
   1. operating_systems.pdf (score: 0.924)
   2. concurrency_chapter.pdf (score: 0.891)

❓ Question: What are the latest research papers on scheduling?

📚 [Arxiv: Matched keywords: latest, research, scheduling]
🔎 Searching Arxiv...
✅ Found 3 relevant papers

📚 Related Research Papers:
==================================================
1. 📄 Efficient Task Scheduling for Edge Computing
   Authors: Zhang et al. (2024)
   🔗 https://arxiv.org/abs/2401.xxxxx
...

📚 Add these papers to the knowledge base? (y/n): y
✅ Papers added to vector database!
```

---

## 🔬 Technical Deep-Dive

### Hierarchical Chunking Strategy
Documents are parsed into three levels for optimal retrieval:
- **Level 1 (2048 tokens)**: Major sections — preserves high-level structure
- **Level 2 (512 tokens)**: Paragraphs — captures topic coherence
- **Level 3 (128 tokens)**: Sentences — enables precise retrieval

The auto-merging retriever dynamically combines chunks when related information spans multiple nodes.

### Semantic Caching Algorithm
1. Compute embedding for incoming query using all-mpnet-base-v2
2. Compare against cached query embeddings using cosine similarity
3. If similarity > 92%, return cached response instantly
4. Otherwise, execute full RAG pipeline and cache result

### Fact Verification Pipeline
1. Collect last 5 Q&A pairs from current session
2. Send to Gemini 1.5 Flash with structured verification prompt
3. Parse response for accuracy scores and detected conflicts
4. Optionally apply corrections to vector database

### Arxiv Integration
1. Extract key terms from user query (filtering stop words)
2. Build Arxiv API query with category filters (cs.OS, cs.DC, etc.)
3. Fetch and display relevant papers with abstracts
4. Optionally add paper abstracts to vector database for future queries

---

## ⚙️ Configuration

Key settings in `config.py`:

```python
# Operation Mode ("local" or "self-learning")
OPERATION_MODE = "self-learning"

# LM Studio Configuration
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL = "qwen3-4b-thinking-2507"

# Verification Settings
VERIFICATION_QUERY_INTERVAL = 5  # Queries between Gemini checks

# Caching
SEMANTIC_SIMILARITY_THRESHOLD = 0.92  # Cache hit threshold

# Arxiv Categories
ARXIV_CATEGORIES = ["cs.OS", "cs.DC", "cs.PF", "cs.AR", "cs.NI"]
```

---

## 📝 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built with these excellent open-source projects:
- [LlamaIndex](https://github.com/run-llama/llama_index) — RAG framework
- [ChromaDB](https://www.trychroma.com/) — Vector database
- [LM Studio](https://lmstudio.ai/) — Local LLM runtime
- [HuggingFace Transformers](https://huggingface.co/) — Embeddings & reranking
- [Google Gemini](https://ai.google.dev/) — Fact verification
- [Arxiv](https://arxiv.org/) — Research paper access

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

*Intelligent Document Understanding • Conversational Memory • Self-Learning Knowledge Base*

**[Quick Start](#-quick-start) • [Features](#-features) • [Commands](#-command-reference)**

Made with ❤️ by [iDheer](https://github.com/iDheer)

</div>
