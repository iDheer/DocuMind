"""
DocuMind Analytics & Feedback Module
=====================================
Tracks query patterns, response quality, and provides insights.
"""

import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict

import config


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@dataclass
class QueryRecord:
    """Record of a single query."""
    query: str
    response_length: int
    sources_count: int
    timestamp: str
    session_id: str
    cache_hit: bool = False
    rating: Optional[int] = None  # 1-5 stars
    feedback: Optional[str] = None
    arxiv_papers_count: int = 0
    verification_triggered: bool = False


@dataclass
class SourceUsage:
    """Track which documents are used most frequently."""
    file_name: str
    usage_count: int
    last_used: str
    avg_relevance_score: float


class QueryAnalytics:
    """
    Tracks and analyzes query patterns and system usage.
    
    Features:
    - Query frequency and patterns
    - Response quality metrics
    - Source document usage tracking
    - User feedback collection
    - Performance insights
    """
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path or os.path.join(config.CHAT_HISTORY_PATH, "analytics")
        os.makedirs(self.data_path, exist_ok=True)
        
        self.queries_file = os.path.join(self.data_path, "queries.json")
        self.sources_file = os.path.join(self.data_path, "sources.json")
        
        self.queries: List[QueryRecord] = []
        self.source_usage: Dict[str, SourceUsage] = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load analytics data from disk."""
        if os.path.exists(self.queries_file):
            try:
                with open(self.queries_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.queries = [QueryRecord(**q) for q in data]
            except Exception as e:
                print(f"⚠️ Failed to load queries: {e}")
                self.queries = []
        
        if os.path.exists(self.sources_file):
            try:
                with open(self.sources_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.source_usage = {
                        k: SourceUsage(**v) for k, v in data.items()
                    }
            except Exception as e:
                print(f"⚠️ Failed to load sources: {e}")
                self.source_usage = {}
    
    def _save_data(self):
        """Save analytics data to disk."""
        try:
            queries_data = [sanitize_for_json(asdict(q)) for q in self.queries]
            with open(self.queries_file, "w", encoding="utf-8") as f:
                json.dump(queries_data, f, indent=2)
            
            sources_data = {k: sanitize_for_json(asdict(v)) for k, v in self.source_usage.items()}
            with open(self.sources_file, "w", encoding="utf-8") as f:
                json.dump(sources_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save analytics: {e}")
    
    def record_query(
        self,
        query: str,
        response: str,
        sources: List[Dict],
        session_id: str,
        cache_hit: bool = False,
        arxiv_papers_count: int = 0,
        verification_triggered: bool = False
    ) -> QueryRecord:
        """
        Record a query for analytics.
        
        Args:
            query: The user's query
            response: The system's response
            sources: List of source documents used
            session_id: Current session ID
            cache_hit: Whether this was a cache hit
            arxiv_papers_count: Number of Arxiv papers fetched
            verification_triggered: Whether Gemini verification ran
            
        Returns:
            The created QueryRecord
        """
        record = QueryRecord(
            query=query,
            response_length=len(response),
            sources_count=len(sources),
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            cache_hit=cache_hit,
            arxiv_papers_count=arxiv_papers_count,
            verification_triggered=verification_triggered
        )
        
        self.queries.append(record)
        
        # Update source usage
        for src in sources:
            file_name = src.get("file", "unknown")
            score = float(src.get("score", 0.0))  # Convert numpy float to Python float
            
            if file_name in self.source_usage:
                usage = self.source_usage[file_name]
                # Update rolling average
                new_count = usage.usage_count + 1
                usage.avg_relevance_score = (
                    (usage.avg_relevance_score * usage.usage_count + score) / new_count
                )
                usage.usage_count = new_count
                usage.last_used = record.timestamp
            else:
                self.source_usage[file_name] = SourceUsage(
                    file_name=file_name,
                    usage_count=1,
                    last_used=record.timestamp,
                    avg_relevance_score=score
                )
        
        self._save_data()
        return record
    
    def add_feedback(
        self, 
        rating: int, 
        feedback: Optional[str] = None
    ) -> bool:
        """
        Add feedback to the most recent query.
        
        Args:
            rating: 1-5 star rating
            feedback: Optional text feedback
            
        Returns:
            True if feedback was added successfully
        """
        if not self.queries:
            return False
        
        if not 1 <= rating <= 5:
            print("⚠️ Rating must be 1-5")
            return False
        
        self.queries[-1].rating = rating
        self.queries[-1].feedback = feedback
        self._save_data()
        return True
    
    def get_stats(self, days: int = 30) -> Dict:
        """
        Get analytics statistics for the last N days.
        
        Args:
            days: Number of days to include
            
        Returns:
            Dictionary of statistics
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent = [
            q for q in self.queries 
            if datetime.fromisoformat(q.timestamp) > cutoff
        ]
        
        if not recent:
            return {
                "total_queries": 0,
                "period_days": days,
                "message": "No queries in this period"
            }
        
        # Calculate stats
        rated = [q for q in recent if q.rating is not None]
        cache_hits = sum(1 for q in recent if q.cache_hit)
        verification_runs = sum(1 for q in recent if q.verification_triggered)
        arxiv_fetches = sum(q.arxiv_papers_count for q in recent)
        
        return {
            "total_queries": len(recent),
            "period_days": days,
            "cache_hit_rate": cache_hits / len(recent) * 100 if recent else 0,
            "avg_response_length": sum(q.response_length for q in recent) / len(recent),
            "avg_sources_per_query": sum(q.sources_count for q in recent) / len(recent),
            "rated_queries": len(rated),
            "avg_rating": sum(q.rating for q in rated) / len(rated) if rated else None,
            "verification_runs": verification_runs,
            "arxiv_papers_fetched": arxiv_fetches,
            "unique_sessions": len(set(q.session_id for q in recent))
        }
    
    def get_top_sources(self, limit: int = 10) -> List[Tuple[str, int, float]]:
        """
        Get the most frequently used source documents.
        
        Args:
            limit: Max number of sources to return
            
        Returns:
            List of (file_name, usage_count, avg_score) tuples
        """
        sorted_sources = sorted(
            self.source_usage.values(),
            key=lambda x: x.usage_count,
            reverse=True
        )[:limit]
        
        return [
            (s.file_name, s.usage_count, s.avg_relevance_score)
            for s in sorted_sources
        ]
    
    def get_query_patterns(self) -> Dict:
        """
        Analyze query patterns.
        
        Returns:
            Dictionary with pattern insights
        """
        if not self.queries:
            return {"message": "No queries to analyze"}
        
        # Query length distribution
        lengths = [len(q.query) for q in self.queries]
        
        # Common starting words
        first_words = Counter()
        for q in self.queries:
            words = q.query.lower().split()
            if words:
                first_words[words[0]] += 1
        
        # Hourly distribution
        hours = Counter()
        for q in self.queries:
            try:
                dt = datetime.fromisoformat(q.timestamp)
                hours[dt.hour] += 1
            except:
                pass
        
        return {
            "total_queries": len(self.queries),
            "avg_query_length": sum(lengths) / len(lengths),
            "min_query_length": min(lengths),
            "max_query_length": max(lengths),
            "top_first_words": first_words.most_common(10),
            "peak_hours": hours.most_common(5)
        }
    
    def print_dashboard(self):
        """Print a formatted analytics dashboard."""
        stats = self.get_stats()
        patterns = self.get_query_patterns()
        top_sources = self.get_top_sources(5)
        
        print("\n" + "=" * 60)
        print("📊 DocuMind Analytics Dashboard")
        print("=" * 60)
        
        print("\n📈 Usage Statistics (Last 30 Days):")
        print(f"   Total Queries: {stats.get('total_queries', 0)}")
        print(f"   Unique Sessions: {stats.get('unique_sessions', 0)}")
        print(f"   Cache Hit Rate: {stats.get('cache_hit_rate', 0):.1f}%")
        print(f"   Avg Response Length: {stats.get('avg_response_length', 0):.0f} chars")
        print(f"   Avg Sources/Query: {stats.get('avg_sources_per_query', 0):.1f}")
        
        if stats.get('avg_rating'):
            print(f"   Avg Rating: {'⭐' * int(stats['avg_rating'])} ({stats['avg_rating']:.1f}/5)")
        
        print(f"\n🔬 Feature Usage:")
        print(f"   Verification Runs: {stats.get('verification_runs', 0)}")
        print(f"   Arxiv Papers Fetched: {stats.get('arxiv_papers_fetched', 0)}")
        
        if top_sources:
            print(f"\n📄 Top Sources:")
            for i, (name, count, score) in enumerate(top_sources, 1):
                print(f"   {i}. {name[:40]} ({count} uses, avg score: {score:.3f})")
        
        if patterns.get("top_first_words"):
            print(f"\n🔤 Common Query Starters:")
            for word, count in patterns["top_first_words"][:5]:
                print(f"   '{word}': {count} times")
        
        print("\n" + "=" * 60)


class FeedbackCollector:
    """
    Interactive feedback collection for responses.
    """
    
    def __init__(self, analytics: QueryAnalytics):
        self.analytics = analytics
    
    def collect_feedback(self) -> bool:
        """
        Prompt user for feedback on the last response.
        
        Returns:
            True if feedback was collected
        """
        print("\n💭 How was this response?")
        print("   Rate 1-5 (or press Enter to skip):")
        
        try:
            rating_input = input("   Rating: ").strip()
            if not rating_input:
                return False
            
            rating = int(rating_input)
            if not 1 <= rating <= 5:
                print("   ⚠️ Please enter 1-5")
                return False
            
            feedback = None
            if rating <= 2:
                feedback = input("   What could be improved? ").strip()
            
            self.analytics.add_feedback(rating, feedback)
            print(f"   ✅ Thanks for your feedback! {'⭐' * rating}")
            return True
            
        except ValueError:
            print("   ⚠️ Invalid rating")
            return False
        except KeyboardInterrupt:
            return False
