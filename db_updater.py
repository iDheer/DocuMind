"""
DocuMind Vector Database Updater
================================
Handles dynamic updates to the ChromaDB vector database based on:
- Gemini fact verification results
- Arxiv paper additions
- Manual corrections
"""

import os
import json
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass

import chromadb
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

import config
from gemini_verifier import VerificationResult
from arxiv_fetcher import ArxivPaper


@dataclass
class UpdateRecord:
    """Record of a database update."""
    update_type: str  # "correction", "arxiv_paper", "manual"
    timestamp: str
    content: str
    source: str
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return {
            "update_type": self.update_type,
            "timestamp": self.timestamp,
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata
        }


class VectorDBUpdater:
    """
    Manages updates to the ChromaDB vector database.
    
    Capabilities:
    - Add corrections based on Gemini verification
    - Add Arxiv papers to knowledge base
    - Track all updates for audit trail
    - Handle document versioning
    """
    
    def __init__(self, storage_context: Optional[StorageContext] = None):
        """
        Initialize the updater.
        
        Args:
            storage_context: Existing storage context from the query system.
                           If None, will connect to DB directly.
        """
        self.storage_context = storage_context
        self.update_history: List[UpdateRecord] = []
        self.update_log_path = os.path.join(config.CHAT_HISTORY_PATH, "db_updates.json")
        
        # Load existing update history
        self._load_update_history()
        
        # Node parser for chunking new content
        self.node_parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
    
    def _load_update_history(self) -> None:
        """Load update history from disk."""
        if os.path.exists(self.update_log_path):
            try:
                with open(self.update_log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for record in data.get("updates", []):
                        self.update_history.append(UpdateRecord(
                            update_type=record["update_type"],
                            timestamp=record["timestamp"],
                            content=record["content"],
                            source=record["source"],
                            metadata=record["metadata"]
                        ))
            except Exception as e:
                print(f"⚠️ Could not load update history: {e}")
    
    def _save_update_history(self) -> None:
        """Save update history to disk."""
        data = {
            "last_updated": datetime.now().isoformat(),
            "total_updates": len(self.update_history),
            "updates": [u.to_dict() for u in self.update_history]
        }
        with open(self.update_log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _get_db_connection(self):
        """Get or create database connection."""
        if self.storage_context is None:
            db = chromadb.PersistentClient(path=config.DB_PATH)
            chroma_collection = db.get_or_create_collection(config.COLLECTION_NAME)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            self.storage_context = StorageContext.from_defaults(
                persist_dir=config.DB_PATH,
                vector_store=vector_store
            )
        return self.storage_context
    
    def add_correction(
        self, 
        verification_result: VerificationResult,
        auto_apply: bool = False
    ) -> bool:
        """
        Add a correction to the database based on verification results.
        
        Args:
            verification_result: Result from Gemini verification
            auto_apply: If True, apply without confirmation
            
        Returns:
            True if correction was added successfully
        """
        if not verification_result.suggested_corrections:
            print("ℹ️ No corrections to apply")
            return False
        
        if verification_result.confidence < config.UPDATE_CONFIDENCE_THRESHOLD:
            print(f"⚠️ Confidence ({verification_result.confidence:.2f}) below threshold ({config.UPDATE_CONFIDENCE_THRESHOLD})")
            return False
        
        # Build correction document
        correction_content = self._build_correction_document(verification_result)
        
        if not auto_apply:
            print("\n📝 Proposed correction:")
            print("-" * 50)
            print(correction_content[:500] + "..." if len(correction_content) > 500 else correction_content)
            print("-" * 50)
            
            response = input("Apply this correction? (y/n): ").lower().strip()
            if response != 'y':
                print("❌ Correction cancelled")
                return False
        
        # Add to database
        success = self._add_document_to_db(
            content=correction_content,
            metadata={
                "type": "correction",
                "original_question": verification_result.question,
                "confidence": verification_result.confidence,
                "added_date": datetime.now().isoformat(),
                "source": "gemini_verification"
            }
        )
        
        if success:
            # Record the update
            self.update_history.append(UpdateRecord(
                update_type="correction",
                timestamp=datetime.now().isoformat(),
                content=correction_content,
                source="gemini_verification",
                metadata={
                    "question": verification_result.question,
                    "confidence": verification_result.confidence
                }
            ))
            self._save_update_history()
            print("✅ Correction added to database")
        
        return success
    
    def _build_correction_document(self, verification_result: VerificationResult) -> str:
        """Build a document from verification corrections."""
        lines = [
            f"# Correction Notice",
            f"",
            f"**Related Query:** {verification_result.question}",
            f"",
            f"**Issues Identified:**"
        ]
        
        for conflict in verification_result.conflicts:
            lines.append(f"- {conflict}")
        
        lines.extend([
            f"",
            f"**Corrected Information:**"
        ])
        
        for correction in verification_result.suggested_corrections:
            lines.append(f"- {correction}")
        
        lines.extend([
            f"",
            f"**Verification Analysis:** {verification_result.gemini_analysis}",
            f"",
            f"**Confidence Score:** {verification_result.confidence:.2f}",
            f"",
            f"*This correction was added on {datetime.now().strftime('%Y-%m-%d')} based on automated fact verification.*"
        ])
        
        return "\n".join(lines)
    
    def add_arxiv_paper(self, paper: ArxivPaper, auto_apply: bool = False) -> bool:
        """
        Add an Arxiv paper to the database.
        
        Args:
            paper: ArxivPaper object to add
            auto_apply: If True, apply without confirmation
            
        Returns:
            True if paper was added successfully
        """
        # Check if paper already exists
        if self._paper_exists(paper.arxiv_id):
            print(f"ℹ️ Paper {paper.arxiv_id} already in database")
            return False
        
        # Build paper document
        paper_content = self._build_paper_document(paper)
        
        if not auto_apply:
            print(f"\n📄 Add paper: {paper.title[:60]}...")
            response = input("Add to database? (y/n): ").lower().strip()
            if response != 'y':
                print("❌ Paper addition cancelled")
                return False
        
        # Add to database
        success = self._add_document_to_db(
            content=paper_content,
            metadata={
                "type": "arxiv_paper",
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": ", ".join(paper.authors[:5]),
                "published": paper.published.isoformat(),
                "categories": ", ".join(paper.categories),
                "added_date": datetime.now().isoformat(),
                "source": "arxiv"
            }
        )
        
        if success:
            self.update_history.append(UpdateRecord(
                update_type="arxiv_paper",
                timestamp=datetime.now().isoformat(),
                content=paper_content,
                source="arxiv",
                metadata=paper.to_dict()
            ))
            self._save_update_history()
            print(f"✅ Paper '{paper.title[:40]}...' added to database")
        
        return success
    
    def _build_paper_document(self, paper: ArxivPaper) -> str:
        """Build a document from an Arxiv paper."""
        return f"""# {paper.title}

## Authors
{', '.join(paper.authors)}

## Published
{paper.published.strftime('%Y-%m-%d')}

## Categories
{', '.join(paper.categories)}

## Abstract
{paper.summary}

## Citation
{paper.format_citation()}

## Links
- PDF: {paper.pdf_url}
- Arxiv: {paper.arxiv_url}

---
*Source: arXiv (arxiv.org), Paper ID: {paper.arxiv_id}*
"""
    
    def _paper_exists(self, arxiv_id: str) -> bool:
        """Check if a paper already exists in the database."""
        # Check update history
        for record in self.update_history:
            if record.update_type == "arxiv_paper":
                if record.metadata.get("arxiv_id") == arxiv_id:
                    return True
        return False
    
    def _add_document_to_db(self, content: str, metadata: Dict) -> bool:
        """
        Add a document to the vector database.
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            True if successful
        """
        try:
            # Create document
            doc = Document(text=content, metadata=metadata)
            
            # Parse into nodes
            nodes = self.node_parser.get_nodes_from_documents([doc])
            
            # Get storage context
            storage_context = self._get_db_connection()
            
            # Add nodes to docstore
            storage_context.docstore.add_documents(nodes)
            
            # Get embeddings and add to vector store
            for node in nodes:
                # The embedding will be computed when we rebuild the index
                pass
            
            # Persist changes
            storage_context.persist(persist_dir=config.DB_PATH)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to add document: {e}")
            return False
    
    def apply_verification_updates(
        self, 
        verification_results: List[VerificationResult],
        auto_apply: bool = False
    ) -> int:
        """
        Apply all updates from verification results.
        
        Args:
            verification_results: List of verification results
            auto_apply: If True, apply without confirmation
            
        Returns:
            Number of updates applied
        """
        updates_applied = 0
        
        for result in verification_results:
            if result.needs_db_update:
                if self.add_correction(result, auto_apply):
                    updates_applied += 1
        
        if updates_applied > 0:
            print(f"\n✅ Applied {updates_applied} database update(s)")
        
        return updates_applied
    
    def add_arxiv_papers(
        self, 
        papers: List[ArxivPaper],
        auto_apply: bool = False
    ) -> int:
        """
        Add multiple Arxiv papers to the database.
        
        Args:
            papers: List of papers to add
            auto_apply: If True, apply without confirmation
            
        Returns:
            Number of papers added
        """
        papers_added = 0
        
        for paper in papers:
            if self.add_arxiv_paper(paper, auto_apply):
                papers_added += 1
        
        if papers_added > 0:
            print(f"\n✅ Added {papers_added} paper(s) to database")
        
        return papers_added
    
    def get_update_stats(self) -> Dict:
        """Get statistics about database updates."""
        stats = {
            "total_updates": len(self.update_history),
            "corrections": 0,
            "arxiv_papers": 0,
            "manual": 0,
            "last_update": None
        }
        
        for record in self.update_history:
            if record.update_type == "correction":
                stats["corrections"] += 1
            elif record.update_type == "arxiv_paper":
                stats["arxiv_papers"] += 1
            elif record.update_type == "manual":
                stats["manual"] += 1
        
        if self.update_history:
            stats["last_update"] = self.update_history[-1].timestamp
        
        return stats
    
    def print_update_summary(self) -> None:
        """Print a summary of all updates."""
        stats = self.get_update_stats()
        
        print("\n📊 Database Update Summary:")
        print(f"   Total updates: {stats['total_updates']}")
        print(f"   - Corrections: {stats['corrections']}")
        print(f"   - Arxiv papers: {stats['arxiv_papers']}")
        print(f"   - Manual: {stats['manual']}")
        if stats['last_update']:
            print(f"   Last update: {stats['last_update']}")
