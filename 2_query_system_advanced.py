import chromadb
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# --- 0. Define Constants ---
DB_PATH = "./chroma_db_advanced"

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

# --- 2. LOAD THE PERSISTED INDEX (The Definitive, Correct Method) ---
print(f"--- Loading data from persisted storage at '{DB_PATH}' ---")

try:
    # Step A: Connect to the persistent ChromaDB vector store
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_or_create_collection("advanced_docs_v1")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    print("--- Connected to existing ChromaDB vector store ---")

    # Step B: Load the rest of the storage context from disk
    # This loads the docstore and index_store
    storage_context = StorageContext.from_defaults(
        persist_dir=DB_PATH, vector_store=vector_store
    )
    print("--- Loaded docstore and index_store from disk ---")
    
    # Step C: Load the index from the now-complete storage context
    print("--- Loading index from complete storage context ---")
    index = load_index_from_storage(storage_context)

except FileNotFoundError:
    print(f"Error: Storage directory not found at '{DB_PATH}'.")
    print("Please run '1_build_database_advanced.py' first to create the database.")
    exit()

# --- 3. CONFIGURE THE ADVANCED RETRIEVER & QUERY ENGINE ---
print("--- Configuring advanced retriever with AutoMerging and Re-ranking ---")
base_retriever = index.as_retriever(similarity_top_k=12)
retriever = AutoMergingRetriever(
    base_retriever,
    index.storage_context,
    verbose=True
)

reranker = SentenceTransformerRerank(
    top_n=4,
    model="BAAI/bge-reranker-v2-m3",
    device="cuda"
)

query_engine = RetrieverQueryEngine.from_args(
    retriever,
    node_postprocessors=[reranker],
    streaming=True,  # Re-enable streaming for real-time output
)

def print_qwen3_response_stream(response_stream):
    """Custom function to handle Qwen3's thinking tags while streaming"""
    full_text = ""
    in_think_tag = False
    buffer = ""
    
    print("🤖 Response: ", end="", flush=True)
    
    for text in response_stream.response_gen:
        full_text += text
        buffer += text
        
        # Process the buffer character by character for real-time output
        while buffer:
            # Check for think tag opening
            if "<think>" in buffer and not in_think_tag:
                think_start = buffer.find("<think>")
                # Print everything before the think tag
                if think_start > 0:
                    print(buffer[:think_start], end="", flush=True)
                
                in_think_tag = True
                print("\n\n🤔 Thinking: ", end="", flush=True)
                
                # Remove processed part and the opening tag
                buffer = buffer[think_start + 7:]
                continue
            
            # Check for think tag closing
            elif "</think>" in buffer and in_think_tag:
                think_end = buffer.find("</think>")
                # Print the thinking content up to the closing tag
                if think_end > 0:
                    print(buffer[:think_end], end="", flush=True)
                
                in_think_tag = False
                print("\n\n💭 Final Answer: ", end="", flush=True)
                
                # Remove processed part and the closing tag
                buffer = buffer[think_end + 8:]
                continue
            
            # If we're in a complete chunk without tags, print it
            else:
                if not ("<think>" in buffer or "</think>" in buffer):
                    print(buffer, end="", flush=True)
                    buffer = ""
                else:
                    # Wait for more text to complete the tag
                    break
    
    # Print any remaining buffer content
    if buffer:
        print(buffer, end="", flush=True)
    
    print()  # Final newline
    return full_text

# --- 4. ASK QUESTIONS ---
print("\n--- Ready to Query! ---")
print("Ask questions about your documents. Type 'exit' or press Ctrl+C to quit.")

try:
    while True:
        question = input("\nQuestion: ")
        if question.lower() == 'exit':
            break

        response = query_engine.query(question)

        # Use our custom streaming function to handle thinking tags
        print_qwen3_response_stream(response)

        print("\n--- Source Nodes (Post-Reranking) ---")
        for i, node in enumerate(response.source_nodes):
            print(f"Source {i+1} (Score: {node.score:.4f}):")
            cleaned_text = ' '.join(node.get_content().split())
            print(f"  -> File: {node.metadata.get('file_name', 'N/A')}")
            print(f"  -> Content Snippet: \"{cleaned_text[:250]}...\"\n")
        print("--------------------------------------")

except KeyboardInterrupt:
    print("\nExiting gracefully. Goodbye!")