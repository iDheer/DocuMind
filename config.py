"""
DocuMind Configuration
======================
Central configuration for all settings, API keys, and constants.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# OPERATION MODE
# ============================================================================
# Two modes available:
#   - "local"         : Privacy-focused, no online features (Gemini, Arxiv disabled)
#   - "self-learning" : Full features enabled (Gemini verification, Arxiv papers)
#
# Can be overridden by environment variable or changed at runtime
OPERATION_MODE = os.getenv("DOCUMIND_MODE", "self-learning")  # "local" or "self-learning"

# ============================================================================
# API KEYS (Set these in .env file or as environment variables)
# ============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ============================================================================
# DATABASE SETTINGS
# ============================================================================
DB_PATH = "./chroma_db_advanced"
DATA_PATH = "./data_large"
COLLECTION_NAME = "advanced_docs_v1"

# ============================================================================
# MODEL SETTINGS
# ============================================================================
# Local LLM (LM Studio - OpenAI compatible API)
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL = "qwen/qwen3-4b-thinking-2507"  # Qwen 3 4B with thinking/reasoning
LM_STUDIO_TIMEOUT = 300.0

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DEVICE = "cuda"  # "cuda" or "cpu"

# Reranker Model (using smaller model for lower memory usage)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_N = 4
RERANKER_DEVICE = "cuda"

# Gemini Model (for fact verification)
GEMINI_MODEL = "gemini-2.5-flash"  # Fast and cost-effective for verification

# ============================================================================
# RETRIEVAL SETTINGS
# ============================================================================
SIMILARITY_TOP_K = 12  # Initial retrieval count before reranking

# ============================================================================
# CHAT HISTORY SETTINGS
# ============================================================================
# Number of recent messages to keep in full context
CHAT_HISTORY_FULL_CONTEXT = 10

# Number of older messages to summarize into a single context block
CHAT_HISTORY_SUMMARIZE_AFTER = 10

# Maximum tokens for the summarized history
SUMMARY_MAX_TOKENS = 500

# ============================================================================
# GEMINI FACT VERIFICATION SETTINGS
# ============================================================================
# Number of queries after which to trigger Gemini verification
VERIFICATION_QUERY_INTERVAL = 5

# Confidence threshold for updating the database (0.0 - 1.0)
# Higher = more conservative updates
UPDATE_CONFIDENCE_THRESHOLD = 0.7

# ============================================================================
# ARXIV INTEGRATION SETTINGS
# ============================================================================
# Maximum number of papers to fetch per query
ARXIV_MAX_RESULTS = 3

# Whether to add arxiv papers to vector DB or just show as references
ARXIV_ADD_TO_DB = True  # Papers will be added to enhance the knowledge base

# Categories to search (empty = all categories)
# cs.OS = Operating Systems, cs.DC = Distributed Computing, cs.PF = Performance
ARXIV_CATEGORIES = ["cs.OS", "cs.DC", "cs.PF", "cs.AR", "cs.NI"]

# ============================================================================
# STORAGE PATHS
# ============================================================================
CHAT_HISTORY_PATH = "./chat_history"
ARXIV_CACHE_PATH = "./arxiv_cache"

# Create directories if they don't exist
os.makedirs(CHAT_HISTORY_PATH, exist_ok=True)
os.makedirs(ARXIV_CACHE_PATH, exist_ok=True)

# ============================================================================
# MODE HELPER FUNCTIONS
# ============================================================================
def is_local_mode() -> bool:
    """Check if running in local/private mode."""
    return OPERATION_MODE.lower() == "local"

def is_self_learning_mode() -> bool:
    """Check if running in self-learning mode with online features."""
    return OPERATION_MODE.lower() == "self-learning"

def get_mode_description() -> str:
    """Get a description of the current mode."""
    if is_local_mode():
        return "🔒 LOCAL MODE (Privacy-focused, offline only)"
    else:
        return "🧠 SELF-LEARNING MODE (Online features enabled)"

def set_mode(mode: str) -> bool:
    """
    Set the operation mode at runtime.
    
    Args:
        mode: "local" or "self-learning"
        
    Returns:
        True if mode was set successfully
    """
    global OPERATION_MODE
    mode = mode.lower().strip()
    if mode in ["local", "self-learning"]:
        OPERATION_MODE = mode
        return True
    return False
