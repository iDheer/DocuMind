import os
import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- 0. Define Constants ---
DB_PATH = "./chroma_db_advanced"
DATA_PATH = "./data_large"

# --- 1. CONFIGURE MODELS ---
print("--- Configuring models ---")
Settings.llm = OpenAILike(
    model="qwen3-4b-thinking-2507",
    api_base="http://localhost:1234/v1",
    api_key="lm-studio",
    timeout=300.0,
    is_chat_model=True
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2",
    device="cuda"
)

# --- 2. INITIALIZE STORAGE ---
print(f"--- Setting up ChromaDB at {DB_PATH} ---")
db = chromadb.PersistentClient(path=DB_PATH)
chroma_collection = db.get_or_create_collection("advanced_docs_v1")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- 3. LOAD DOCUMENTS ---
print(f"--- Loading documents from '{DATA_PATH}' ---")
if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
    os.makedirs(DATA_PATH, exist_ok=True)
    print(f"Created '{DATA_PATH}' directory. Please add your large documents (e.g., PDFs) there and run again.")
    exit()

documents = SimpleDirectoryReader(DATA_PATH).load_data()
print(f"Loaded {len(documents)} document(s).")

# --- 4. PARSE WITH HIERARCHICAL NODE PARSER ---
print("--- Parsing documents with HierarchicalNodeParser ---")
node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
nodes = node_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(nodes)
print(f"Generated {len(nodes)} total nodes and {len(leaf_nodes)} leaf nodes for embedding.")

# --- 5. BUILD INDEX AND PERSIST DATA (Corrected Method) ---
# First, explicitly add ALL nodes (parents and children) to the document store.
# This is the key fix for the "doc_id not found" error.
storage_context.docstore.add_documents(nodes)
print(f"--- Stored {len(nodes)} total nodes in the docstore ---")

# Then, build the vector index ONLY from the leaf nodes.
print(f"--- Building vector index from {len(leaf_nodes)} leaf nodes... This may take a while. ---")
vector_index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context,
    show_progress=True,
)

# Finally, persist the entire storage context to disk.
# This saves the vector store, docstore, and index store.
print(f"--- Persisting all data to '{DB_PATH}' ---")
vector_index.storage_context.persist(persist_dir=DB_PATH)

print("\n--- Ingestion Complete ---")
print(f"Successfully built and saved advanced index at '{DB_PATH}'")