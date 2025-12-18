"""
DocuMind Gemini Fact Verifier
=============================
Uses Google Gemini API to verify responses, detect conflicts,
and identify outdated information that needs updating.
"""

import json
import os
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import config

# Lazy imports
_gemini_llm = None
_cache_manager = None


def get_gemini_llm():
    """Lazy initialization of Gemini LLM."""
    global _gemini_llm
    if _gemini_llm is None:
        if not config.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not set. Please set it in your .env file or config.py"
            )
        os.environ["GOOGLE_API_KEY"] = config.GEMINI_API_KEY
        from llama_index.llms.gemini import Gemini
        _gemini_llm = Gemini(model=config.GEMINI_MODEL)
    return _gemini_llm


def get_cache_manager():
    """Lazy initialization of cache manager."""
    global _cache_manager
    if _cache_manager is None:
        from cache_manager import CacheManager
        _cache_manager = CacheManager()
    return _cache_manager


@dataclass
class VerificationResult:
    """Result of fact verification for a single Q&A pair."""
    question: str
    original_answer: str
    is_accurate: bool
    confidence: float  # 0.0 to 1.0
    conflicts: List[str]  # List of conflicting facts
    suggested_corrections: List[str]  # Suggested updates
    gemini_analysis: str  # Full analysis from Gemini
    needs_db_update: bool  # Whether the vector DB should be updated
    
    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "original_answer": self.original_answer,
            "is_accurate": self.is_accurate,
            "confidence": self.confidence,
            "conflicts": self.conflicts,
            "suggested_corrections": self.suggested_corrections,
            "gemini_analysis": self.gemini_analysis,
            "needs_db_update": self.needs_db_update
        }


class GeminiFactVerifier:
    """
    Verifies facts in Q&A pairs using Google Gemini.
    
    Strategy:
    1. Takes the last N Q&A pairs from chat history
    2. Sends them to Gemini for fact-checking
    3. Identifies conflicts, outdated info, or inaccuracies
    4. Returns structured results for database updates
    """
    
    def __init__(self):
        self.verification_history: List[VerificationResult] = []
        self.last_verification_time: Optional[datetime] = None
    
    def verify_qa_pairs(
        self, 
        qa_pairs: List[Tuple[str, str]],
        document_context: Optional[str] = None
    ) -> List[VerificationResult]:
        """
        Verify a list of question-answer pairs.
        
        Args:
            qa_pairs: List of (question, answer) tuples
            document_context: Optional context about what documents are in the system
            
        Returns:
            List of VerificationResult objects
        """
        if not qa_pairs:
            return []
        
        print(f"\n🔍 Verifying {len(qa_pairs)} Q&A pairs with Gemini...")
        
        try:
            llm = get_gemini_llm()
        except ValueError as e:
            print(f"⚠️ {e}")
            return []
        
        # Check cache for this verification request
        cache_manager = get_cache_manager()
        prompt = self._build_verification_prompt(qa_pairs, document_context)
        prompt_hash = cache_manager.gemini_cache.hash_qa_pairs(qa_pairs)
        
        cached_result = cache_manager.gemini_cache.get_verification(prompt_hash)
        if cached_result:
            print("💾 [Cache Hit - Using cached verification]")
            results = self._parse_verification_response(qa_pairs, cached_result)
            self._print_verification_summary(results)
            return results
        
        results = []
        
        try:
            response = llm.complete(prompt)
            analysis = str(response).strip()
            
            # Cache the Gemini response
            cache_manager.gemini_cache.cache_verification(prompt_hash, analysis)
            
            # Parse the structured response
            results = self._parse_verification_response(qa_pairs, analysis)
            
            self.verification_history.extend(results)
            self.last_verification_time = datetime.now()
            
            # Print summary
            self._print_verification_summary(results)
            
            return results
            
        except Exception as e:
            print(f"❌ Gemini verification failed: {e}")
            return []
    
    def _build_verification_prompt(
        self, 
        qa_pairs: List[Tuple[str, str]],
        document_context: Optional[str]
    ) -> str:
        """Build the verification prompt for Gemini."""
        
        qa_formatted = ""
        for i, (q, a) in enumerate(qa_pairs, 1):
            # Truncate long answers
            answer_preview = a[:1000] + "..." if len(a) > 1000 else a
            qa_formatted += f"""
--- Q&A Pair {i} ---
Question: {q}
Answer: {answer_preview}
"""
        
        context_info = ""
        if document_context:
            context_info = f"""
The answers were generated from a document database containing:
{document_context}
"""
        
        prompt = f"""You are a fact-checking assistant. Analyze the following Q&A pairs that were generated by a RAG (Retrieval Augmented Generation) system from a document database.

For each Q&A pair, evaluate:
1. **Factual Accuracy**: Is the information correct based on current knowledge?
2. **Potential Conflicts**: Are there any contradictions or outdated information?
3. **Information Gaps**: Is the answer incomplete or missing important context?
4. **Update Recommendations**: Should the source database be updated?
{context_info}
{qa_formatted}

Provide your analysis in the following JSON format:
```json
{{
  "verifications": [
    {{
      "pair_index": 1,
      "is_accurate": true/false,
      "confidence": 0.0-1.0,
      "conflicts": ["list of conflicting facts if any"],
      "suggested_corrections": ["list of corrections or updates needed"],
      "analysis": "Brief explanation of your assessment",
      "needs_db_update": true/false
    }}
  ],
  "overall_assessment": "Summary of the verification results",
  "critical_updates_needed": ["List of critical updates that should be made to the knowledge base"]
}}
```

Be thorough but fair. Only flag issues if you are reasonably confident there's a problem.
Focus especially on:
- Outdated statistics or data
- Recently changed facts (laws, technologies, scientific understanding)
- Common misconceptions that might be in source documents
- Missing important caveats or context"""

        return prompt
    
    def _parse_verification_response(
        self, 
        qa_pairs: List[Tuple[str, str]], 
        analysis: str
    ) -> List[VerificationResult]:
        """Parse Gemini's response into VerificationResult objects."""
        
        results = []
        
        # Try to extract JSON from the response
        try:
            # Find JSON block in response
            json_start = analysis.find("```json")
            json_end = analysis.rfind("```")
            
            if json_start != -1 and json_end > json_start:
                json_str = analysis[json_start + 7:json_end].strip()
                data = json.loads(json_str)
            else:
                # Try parsing the whole thing as JSON
                data = json.loads(analysis)
            
            verifications = data.get("verifications", [])
            
            for v in verifications:
                pair_idx = v.get("pair_index", 1) - 1
                if pair_idx < len(qa_pairs):
                    q, a = qa_pairs[pair_idx]
                    results.append(VerificationResult(
                        question=q,
                        original_answer=a,
                        is_accurate=v.get("is_accurate", True),
                        confidence=v.get("confidence", 0.5),
                        conflicts=v.get("conflicts", []),
                        suggested_corrections=v.get("suggested_corrections", []),
                        gemini_analysis=v.get("analysis", ""),
                        needs_db_update=v.get("needs_db_update", False)
                    ))
                    
        except json.JSONDecodeError:
            # If JSON parsing fails, create basic results with the raw analysis
            print("⚠️ Could not parse structured response, using basic analysis")
            for q, a in qa_pairs:
                results.append(VerificationResult(
                    question=q,
                    original_answer=a,
                    is_accurate=True,  # Assume accurate if we can't parse
                    confidence=0.5,
                    conflicts=[],
                    suggested_corrections=[],
                    gemini_analysis=analysis,
                    needs_db_update=False
                ))
        
        return results
    
    def _print_verification_summary(self, results: List[VerificationResult]) -> None:
        """Print a summary of verification results."""
        if not results:
            return
            
        accurate_count = sum(1 for r in results if r.is_accurate)
        needs_update_count = sum(1 for r in results if r.needs_db_update)
        
        print(f"\n📊 Verification Summary:")
        print(f"   ✅ Accurate: {accurate_count}/{len(results)}")
        print(f"   ⚠️ Needs attention: {len(results) - accurate_count}/{len(results)}")
        print(f"   🔄 DB updates recommended: {needs_update_count}")
        
        # Print details for items needing attention
        for r in results:
            if not r.is_accurate or r.needs_db_update:
                print(f"\n   📌 Issue detected:")
                print(f"      Q: {r.question[:100]}...")
                if r.conflicts:
                    print(f"      Conflicts: {', '.join(r.conflicts[:2])}")
                if r.suggested_corrections:
                    print(f"      Suggestion: {r.suggested_corrections[0][:100]}...")
    
    def get_updates_for_db(self) -> List[Dict]:
        """
        Get a list of updates that should be applied to the vector database.
        Only returns high-confidence corrections.
        """
        updates = []
        
        for result in self.verification_history:
            if (result.needs_db_update and 
                result.confidence >= config.UPDATE_CONFIDENCE_THRESHOLD):
                updates.append({
                    "question": result.question,
                    "original_answer": result.original_answer,
                    "corrections": result.suggested_corrections,
                    "conflicts": result.conflicts,
                    "confidence": result.confidence
                })
        
        return updates
    
    def save_verification_log(self) -> str:
        """Save verification history to a log file."""
        if not self.verification_history:
            return ""
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(config.CHAT_HISTORY_PATH, f"verification_log_{timestamp}.json")
        
        log_data = {
            "timestamp": timestamp,
            "results": [r.to_dict() for r in self.verification_history],
            "updates_recommended": self.get_updates_for_db()
        }
        
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Verification log saved to: {log_path}")
        return log_path
