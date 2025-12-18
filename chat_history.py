"""
DocuMind Chat History Manager
=============================
Manages conversation history with intelligent summarization to maintain context
while staying within token limits.
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from llama_index.core import Settings

import config


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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
class Message:
    """Represents a single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    sources: Optional[List[Dict]] = None  # Source nodes for assistant responses
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        return cls(**data)


class ChatHistory:
    """
    Manages conversation history with automatic summarization.
    
    Strategy:
    - Keep recent N messages in full detail
    - Summarize older messages into a rolling summary
    - Persist history to disk for session continuity
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize chat history.
        
        Args:
            session_id: Unique identifier for this chat session.
                       If None, creates a new session.
        """
        self.session_id = session_id or self._generate_session_id()
        self.messages: List[Message] = []
        self.summary: str = ""  # Rolling summary of older messages
        self.query_count: int = 0  # Track queries for Gemini verification
        
        # Configuration
        self.full_context_size = config.CHAT_HISTORY_FULL_CONTEXT
        self.summarize_after = config.CHAT_HISTORY_SUMMARIZE_AFTER
        
        # Load existing session if available
        self._load_session()
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID based on timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _get_session_path(self) -> str:
        """Get the file path for this session's history."""
        return os.path.join(config.CHAT_HISTORY_PATH, f"session_{self.session_id}.json")
    
    def _load_session(self) -> None:
        """Load existing session from disk if available."""
        path = self._get_session_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.messages = [Message.from_dict(m) for m in data.get("messages", [])]
                    self.summary = data.get("summary", "")
                    self.query_count = data.get("query_count", 0)
                    print(f"📂 Loaded session '{self.session_id}' with {len(self.messages)} messages")
            except Exception as e:
                print(f"⚠️ Could not load session: {e}")
    
    def save_session(self) -> None:
        """Persist current session to disk."""
        path = self._get_session_path()
        data = {
            "session_id": self.session_id,
            "messages": [sanitize_for_json(m.to_dict()) for m in self.messages],
            "summary": self.summary,
            "query_count": self.query_count,
            "last_updated": datetime.now().isoformat()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self.messages.append(Message(
            role="user",
            content=content,
            timestamp=datetime.now().isoformat()
        ))
        self.query_count += 1
        self._maybe_summarize()
        self.save_session()
    
    def add_assistant_message(self, content: str, sources: Optional[List[Dict]] = None) -> None:
        """Add an assistant response to history."""
        self.messages.append(Message(
            role="assistant",
            content=content,
            timestamp=datetime.now().isoformat(),
            sources=sources
        ))
        self.save_session()
    
    def _maybe_summarize(self) -> None:
        """
        Summarize older messages if we've exceeded the threshold.
        Uses the configured LLM to create a concise summary.
        """
        if len(self.messages) <= self.full_context_size + self.summarize_after:
            return
        
        # Messages to summarize (older ones beyond full context)
        messages_to_summarize = self.messages[:len(self.messages) - self.full_context_size]
        
        if len(messages_to_summarize) < self.summarize_after:
            return
        
        print("📝 Summarizing older conversation history...")
        
        # Build summary prompt
        conversation_text = self._format_messages_for_summary(messages_to_summarize)
        
        summary_prompt = f"""Summarize the following conversation between a user and an AI assistant.
Focus on:
1. Key topics discussed
2. Important facts or information shared
3. Any conclusions or answers provided
4. User's main interests or questions

Keep the summary concise but preserve important context.

Previous Summary (if any):
{self.summary if self.summary else "None"}

Conversation to summarize:
{conversation_text}

Provide a concise summary (max {config.SUMMARY_MAX_TOKENS} words):"""

        try:
            # Use the configured LLM to summarize
            response = Settings.llm.complete(summary_prompt)
            new_summary = str(response).strip()
            
            # Clean up Qwen3 thinking tags if present
            if "<think>" in new_summary and "</think>" in new_summary:
                # Extract content after thinking
                think_end = new_summary.find("</think>")
                new_summary = new_summary[think_end + 8:].strip()
            
            self.summary = new_summary
            
            # Remove summarized messages, keep only recent ones
            self.messages = self.messages[len(messages_to_summarize):]
            
            print(f"✅ Summarized {len(messages_to_summarize)} messages into rolling context")
            
        except Exception as e:
            print(f"⚠️ Summarization failed: {e}")
    
    def _format_messages_for_summary(self, messages: List[Message]) -> str:
        """Format messages for the summarization prompt."""
        lines = []
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            # Truncate long messages for summary
            content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    
    def get_context_for_query(self) -> str:
        """
        Get formatted context string for the current query.
        Includes rolling summary + recent messages.
        """
        context_parts = []
        
        # Add rolling summary if exists
        if self.summary:
            context_parts.append(f"[Previous Conversation Summary]\n{self.summary}\n")
        
        # Add recent messages
        if self.messages:
            context_parts.append("[Recent Conversation]")
            for msg in self.messages[-self.full_context_size:]:
                role = "User" if msg.role == "user" else "Assistant"
                context_parts.append(f"{role}: {msg.content}")
        
        return "\n".join(context_parts)
    
    def get_recent_qa_pairs(self, n: int = 5) -> List[Tuple[str, str]]:
        """
        Get the last N question-answer pairs.
        Used for Gemini verification.
        """
        pairs = []
        i = len(self.messages) - 1
        
        while i > 0 and len(pairs) < n:
            if self.messages[i].role == "assistant" and self.messages[i-1].role == "user":
                pairs.append((
                    self.messages[i-1].content,  # Question
                    self.messages[i].content      # Answer
                ))
                i -= 2
            else:
                i -= 1
        
        return list(reversed(pairs))  # Return in chronological order
    
    def should_verify(self) -> bool:
        """Check if we should trigger Gemini verification based on query count."""
        return self.query_count > 0 and self.query_count % config.VERIFICATION_QUERY_INTERVAL == 0
    
    def clear_history(self) -> None:
        """Clear all history for this session."""
        self.messages = []
        self.summary = ""
        self.query_count = 0
        self.save_session()
        print("🗑️ Chat history cleared")
    
    def get_all_sessions(self) -> List[str]:
        """Get list of all available session IDs."""
        sessions = []
        for filename in os.listdir(config.CHAT_HISTORY_PATH):
            if filename.startswith("session_") and filename.endswith(".json"):
                session_id = filename[8:-5]  # Remove "session_" prefix and ".json" suffix
                sessions.append(session_id)
        return sorted(sessions, reverse=True)  # Most recent first
    
    def __len__(self) -> int:
        return len(self.messages)
    
    def __repr__(self) -> str:
        return f"ChatHistory(session={self.session_id}, messages={len(self.messages)}, queries={self.query_count})"
