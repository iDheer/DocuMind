"""
DocuMind Caching System
=======================
Provides intelligent caching for:
- Query results (semantic similarity-based)
- Embeddings
- Gemini verification results
- Arxiv paper searches

Uses disk-based persistence with TTL (time-to-live) support.
"""

import os
import json
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np

import config


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""
    key: str
    value: Any
    created_at: str
    expires_at: Optional[str]
    hit_count: int = 0
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > datetime.fromisoformat(self.expires_at)


class DiskCache:
    """
    Persistent disk-based cache with TTL support.
    """
    
    def __init__(self, cache_name: str, ttl_hours: int = 24):
        """
        Initialize disk cache.
        
        Args:
            cache_name: Name of the cache (creates subdirectory)
            ttl_hours: Time-to-live in hours (0 = never expire)
        """
        self.cache_dir = os.path.join(config.CHAT_HISTORY_PATH, "cache", cache_name)
        self.ttl_hours = ttl_hours
        self.stats = {"hits": 0, "misses": 0}
        
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_stats()
    
    def _get_cache_path(self, key: str) -> str:
        """Get file path for a cache key."""
        # Hash the key to create a valid filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
    
    def _load_stats(self) -> None:
        """Load cache statistics."""
        stats_path = os.path.join(self.cache_dir, "_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                self.stats = json.load(f)
    
    def _save_stats(self) -> None:
        """Save cache statistics."""
        stats_path = os.path.join(self.cache_dir, "_stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.stats, f)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Returns:
            Cached value or None if not found/expired
        """
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            self.stats["misses"] += 1
            return None
        
        try:
            with open(cache_path, "rb") as f:
                entry: CacheEntry = pickle.load(f)
            
            if entry.is_expired():
                os.remove(cache_path)
                self.stats["misses"] += 1
                return None
            
            # Update hit count
            entry.hit_count += 1
            with open(cache_path, "wb") as f:
                pickle.dump(entry, f)
            
            self.stats["hits"] += 1
            self._save_stats()
            return entry.value
            
        except Exception:
            self.stats["misses"] += 1
            return None
    
    def set(self, key: str, value: Any, ttl_hours: Optional[int] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_hours: Override default TTL
        """
        ttl = ttl_hours if ttl_hours is not None else self.ttl_hours
        
        expires_at = None
        if ttl > 0:
            expires_at = (datetime.now() + timedelta(hours=ttl)).isoformat()
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now().isoformat(),
            expires_at=expires_at,
            hit_count=0
        )
        
        cache_path = self._get_cache_path(key)
        with open(cache_path, "wb") as f:
            pickle.dump(entry, f)
    
    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            return True
        return False
    
    def clear(self) -> int:
        """Clear all cache entries. Returns number of entries cleared."""
        count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".cache"):
                os.remove(os.path.join(self.cache_dir, filename))
                count += 1
        self.stats = {"hits": 0, "misses": 0}
        self._save_stats()
        return count
    
    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns number removed."""
        count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".cache"):
                cache_path = os.path.join(self.cache_dir, filename)
                try:
                    with open(cache_path, "rb") as f:
                        entry: CacheEntry = pickle.load(f)
                    if entry.is_expired():
                        os.remove(cache_path)
                        count += 1
                except Exception:
                    pass
        return count
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        
        # Count entries
        entry_count = len([f for f in os.listdir(self.cache_dir) if f.endswith(".cache")])
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": f"{hit_rate:.1%}",
            "entries": entry_count
        }


class SemanticQueryCache:
    """
    Caches query results with semantic similarity matching.
    If a similar question was asked before, returns cached answer.
    """
    
    def __init__(self, similarity_threshold: float = 0.92):
        """
        Initialize semantic cache.
        
        Args:
            similarity_threshold: Cosine similarity threshold for cache hits (0.0-1.0)
        """
        self.cache = DiskCache("semantic_queries", ttl_hours=72)  # 3 days
        self.similarity_threshold = similarity_threshold
        self.embeddings_cache: Dict[str, List[float]] = {}
        self._load_embeddings_index()
    
    def _get_embeddings_path(self) -> str:
        return os.path.join(self.cache.cache_dir, "_embeddings_index.pkl")
    
    def _load_embeddings_index(self) -> None:
        """Load the embeddings index from disk."""
        path = self._get_embeddings_path()
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.embeddings_cache = pickle.load(f)
            except Exception:
                self.embeddings_cache = {}
    
    def _save_embeddings_index(self) -> None:
        """Save embeddings index to disk."""
        path = self._get_embeddings_path()
        with open(path, "wb") as f:
            pickle.dump(self.embeddings_cache, f)
    
    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for text using the configured model."""
        from llama_index.core import Settings
        embedding = Settings.embed_model.get_text_embedding(text)
        return embedding
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_np = np.array(a)
        b_np = np.array(b)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))
    
    def find_similar(self, query: str) -> Optional[Tuple[str, str, float]]:
        """
        Find a semantically similar cached query.
        
        Returns:
            Tuple of (cached_query, cached_answer, similarity_score) or None
        """
        if not self.embeddings_cache:
            return None
        
        query_embedding = self._compute_embedding(query)
        
        best_match = None
        best_score = 0.0
        
        for cached_query, cached_embedding in self.embeddings_cache.items():
            score = self._cosine_similarity(query_embedding, cached_embedding)
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_match = cached_query
        
        if best_match:
            cached_answer = self.cache.get(best_match)
            if cached_answer:
                return (best_match, cached_answer, best_score)
        
        return None
    
    def cache_query(self, query: str, answer: str) -> None:
        """Cache a query-answer pair with its embedding."""
        # Store the answer
        self.cache.set(query, answer)
        
        # Store the embedding
        embedding = self._compute_embedding(query)
        self.embeddings_cache[query] = embedding
        self._save_embeddings_index()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        stats = self.cache.get_stats()
        stats["cached_queries"] = len(self.embeddings_cache)
        stats["similarity_threshold"] = self.similarity_threshold
        return stats


class GeminiResponseCache:
    """Cache for Gemini API responses to reduce API costs."""
    
    def __init__(self):
        self.cache = DiskCache("gemini_responses", ttl_hours=168)  # 7 days
    
    def get_verification(self, qa_pairs_hash: str) -> Optional[Dict]:
        """Get cached verification result."""
        return self.cache.get(qa_pairs_hash)
    
    def cache_verification(self, qa_pairs_hash: str, result: Dict) -> None:
        """Cache a verification result."""
        self.cache.set(qa_pairs_hash, result)
    
    @staticmethod
    def hash_qa_pairs(qa_pairs: List[Tuple[str, str]]) -> str:
        """Create a hash of Q&A pairs for cache key."""
        content = json.dumps(qa_pairs, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()


class CacheManager:
    """
    Central manager for all caches.
    Provides unified interface and statistics.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.query_cache = SemanticQueryCache(similarity_threshold=0.92)
        self.gemini_cache = GeminiResponseCache()
        self.arxiv_cache = DiskCache("arxiv_papers", ttl_hours=168)  # 7 days
        self._initialized = True
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all caches."""
        return {
            "query_cache": self.query_cache.get_stats(),
            "gemini_cache": self.gemini_cache.cache.get_stats(),
            "arxiv_cache": self.arxiv_cache.get_stats()
        }
    
    def cleanup_all(self) -> Dict[str, int]:
        """Cleanup expired entries from all caches."""
        return {
            "query_cache": self.query_cache.cache.cleanup_expired(),
            "gemini_cache": self.gemini_cache.cache.cleanup_expired(),
            "arxiv_cache": self.arxiv_cache.cleanup_expired()
        }
    
    def clear_all(self) -> Dict[str, int]:
        """Clear all caches."""
        return {
            "query_cache": self.query_cache.cache.clear(),
            "gemini_cache": self.gemini_cache.cache.clear(),
            "arxiv_cache": self.arxiv_cache.clear()
        }
    
    def print_stats(self) -> None:
        """Print cache statistics."""
        stats = self.get_all_stats()
        
        print("\n📊 Cache Statistics:")
        print("-" * 40)
        
        print("\n🔍 Query Cache (Semantic):")
        qs = stats["query_cache"]
        print(f"   Cached queries: {qs['cached_queries']}")
        print(f"   Hit rate: {qs['hit_rate']}")
        print(f"   Hits/Misses: {qs['hits']}/{qs['misses']}")
        
        print("\n🤖 Gemini Response Cache:")
        gs = stats["gemini_cache"]
        print(f"   Cached responses: {gs['entries']}")
        print(f"   Hit rate: {gs['hit_rate']}")
        
        print("\n📚 Arxiv Cache:")
        arxiv_s = stats["arxiv_cache"]
        print(f"   Cached searches: {arxiv_s['entries']}")
        print(f"   Hit rate: {arxiv_s['hit_rate']}")
        
        print("-" * 40)


# Singleton instance
def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    return CacheManager()
