# 🧠 DocuMind — Smart Document Q&A with Hierarchical RAG

> **"Ask your documents anything."** — A powerful, GPU-accelerated RAG system that understands context like never before.

DocuMind is an advanced Retrieval-Augmented Generation (RAG) system that goes beyond simple keyword matching. Using **hierarchical document chunking**, **auto-merging retrieval**, and **neural re-ranking**, it delivers precise, context-aware answers from your PDF documents.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Powered by LlamaIndex](https://img.shields.io/badge/Powered%20by-LlamaIndex-orange.svg)](https://github.com/run-llama/llama_index)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🏛️ **Hierarchical Chunking** | Multi-level document parsing (2048 → 512 → 128 tokens) preserves context hierarchy |
| 🔄 **Auto-Merging Retrieval** | Intelligently combines related chunks for comprehensive answers |
| 🎯 **Neural Re-ranking** | BGE-Reranker-v2-M3 ensures only the most relevant passages reach the LLM |
| ⚡ **Real-time Streaming** | Watch the AI think and respond in real-time with Qwen3's reasoning tags |
| 🚀 **GPU Acceleration** | CUDA-optimized embeddings and reranking for blazing-fast performance |
| 💾 **Persistent Storage** | ChromaDB vector store preserves your processed documents |

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documents     │───▶│  Hierarchical   │───▶│   ChromaDB      │
│   (PDF, etc.)   │    │   Processing    │    │ Vector Store    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Query UI      │◄───│ Auto-Merging    │◄───│   Retriever     │
│  (Terminal)     │    │   + Reranking   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Ollama installed and running
- At least 8GB RAM (16GB+ recommended)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/iDheer/DocuMind.git
   cd DocuMind
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv environment
   # On Windows
   .\environment\Scripts\activate
   # On Linux/Mac
   source environment/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and setup Ollama**
   - Download and install Ollama from [https://ollama.ai](https://ollama.ai)
   - Pull the required model:
     ```bash
     ollama pull qwen3:4b
     ```

## 📁 Project Structure

```
DocuMind/
├── 1_build_database_advanced.py    # 📥 Database builder & document processor
├── 2_query_system_advanced.py      # 💬 Interactive query interface
├── 3_inspect_hierarchy.py          # 🔍 Document hierarchy visualizer
├── requirements.txt                # 📦 Python dependencies
├── README.md                       # 📖 You are here!
├── LICENSE                         # 📜 MIT License
├── data_large/                     # 📂 Your PDF documents (create this)
│   └── *.pdf                       
└── chroma_db_advanced/             # 🗃️ Vector database (auto-generated)
```

## 🚀 Quick Start

### Step 1: Prepare Your Documents
Place your PDF documents in the `data_large/` directory.

### Step 2: Build the Database
```bash
python 1_build_database_advanced.py
```
This script will:
- Process your documents with hierarchical chunking
- Generate embeddings using HuggingFace models
- Store everything in ChromaDB for fast retrieval

### Step 3: Start Querying
```bash
python 2_query_system_advanced.py
```
This launches an interactive terminal interface where you can:
- Ask questions about your documents
- See real-time thinking process from the AI
- View source references and confidence scores

### Step 4: (Optional) Inspect Document Hierarchy
```bash
python 3_inspect_hierarchy.py
```
Use this to understand how your documents were processed and chunked.

## 🎯 Usage Examples

Once you run `2_query_system_advanced.py`, you can ask questions like:

```
Question: What are the main concepts of operating systems?

🤖 Response: 
🤔 Thinking: Let me search through the operating systems documentation to find the main concepts...

💭 Final Answer: Based on the documentation, the main concepts of operating systems include:

1. **Virtualization**: The OS provides abstractions of physical resources
2. **Concurrency**: Managing multiple processes simultaneously
3. **Persistence**: Storing data reliably on storage devices
...
```

## ⚙️ Configuration

### Models Used
- **LLM**: Qwen3:4b (via Ollama)
- **Embeddings**: sentence-transformers/all-mpnet-base-v2
- **Reranker**: BAAI/bge-reranker-v2-m3

### Key Parameters
```python
# In 1_build_database_advanced.py - HierarchicalNodeParser
chunk_sizes = [2048, 512, 128]  # Multi-level chunking hierarchy

# In 2_query_system_advanced.py - Retriever & Reranker
similarity_top_k = 12          # Initial retrieval count
top_n = 4                      # Final reranked results
```

## 🔧 Customization

### Adding New Document Types
Modify `1_build_database_advanced.py` to support additional file formats:
```python
# Add new loaders in the document loading section
from llama_index.readers.file import DocxReader
# ... add your loader logic
```

### Changing Models
Update the model configurations in both scripts:
```python
# For different LLM
Settings.llm = Ollama(model="your-model:tag")

# For different embeddings
Settings.embed_model = HuggingFaceEmbedding(
    model_name="your-embedding-model"
)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Check if CUDA is available in Python
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Switch to CPU if needed - edit the scripts:
   device="cpu"  # in model configurations
   ```

2. **Ollama Connection Issues**
   ```bash
   # Ensure Ollama is running
   ollama serve
   
   # Check if model is available
   ollama list
   ```

3. **ChromaDB Permission Issues**
   ```bash
   # Delete and rebuild database
   rm -rf chroma_db_advanced/
   python 1_build_database_advanced.py
   ```

### Performance Optimization

- **GPU Memory**: Reduce `similarity_top_k` if running out of memory
- **Speed**: Use smaller embedding models for faster processing
- **Quality**: Increase `chunk_overlap` for better context preservation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LlamaIndex](https://github.com/run-llama/llama_index) for the RAG framework
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- [Ollama](https://ollama.ai/) for local LLM inference
- [HuggingFace](https://huggingface.co/) for transformer models

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed information about your problem

---

<div align="center">

**DocuMind** — Built with ❤️ using LlamaIndex, ChromaDB, and Ollama

*Transform your documents into conversational knowledge.*

</div>
