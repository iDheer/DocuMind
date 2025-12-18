"""
DocuMind Arxiv Fetcher
======================
Fetches relevant research papers from Arxiv based on user queries.
Can optionally add papers to the vector database.
"""

import os
import re
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

import arxiv

import config


@dataclass
class ArxivPaper:
    """Represents an Arxiv paper."""
    arxiv_id: str
    title: str
    authors: List[str]
    summary: str
    published: datetime
    updated: datetime
    categories: List[str]
    pdf_url: str
    arxiv_url: str
    
    def to_dict(self) -> Dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "summary": self.summary,
            "published": self.published.isoformat(),
            "updated": self.updated.isoformat(),
            "categories": self.categories,
            "pdf_url": self.pdf_url,
            "arxiv_url": self.arxiv_url
        }
    
    def format_citation(self) -> str:
        """Format paper as a citation."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        year = self.published.year
        return f"{authors_str} ({year}). {self.title}. arXiv:{self.arxiv_id}"
    
    def format_for_display(self) -> str:
        """Format paper for display to user."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += f" et al. (+{len(self.authors) - 3} more)"
        
        # Clean and truncate summary
        summary = re.sub(r'\s+', ' ', self.summary).strip()
        if len(summary) > 300:
            summary = summary[:300] + "..."
        
        return f"""
📄 **{self.title}**
   Authors: {authors_str}
   Published: {self.published.strftime('%Y-%m-%d')}
   Categories: {', '.join(self.categories[:3])}
   
   📝 Abstract: {summary}
   
   🔗 PDF: {self.pdf_url}
   🔗 Arxiv: {self.arxiv_url}
"""


class ArxivFetcher:
    """
    Fetches relevant research papers from Arxiv.
    
    Features:
    - Query-based paper search
    - Smart keyword extraction
    - Caching to avoid redundant API calls
    - Optional integration with vector database
    """
    
    def __init__(self):
        self.cache: Dict[str, List[ArxivPaper]] = {}
        self.client = arxiv.Client(
            page_size=config.ARXIV_MAX_RESULTS,
            delay_seconds=1.0,  # Be nice to the API
            num_retries=3
        )
    
    def search_papers(
        self, 
        query: str, 
        max_results: Optional[int] = None,
        categories: Optional[List[str]] = None
    ) -> List[ArxivPaper]:
        """
        Search Arxiv for papers related to the query.
        
        Args:
            query: Search query (can be natural language)
            max_results: Maximum number of papers to return
            categories: Arxiv categories to filter by (e.g., ["cs.AI", "cs.LG"])
            
        Returns:
            List of ArxivPaper objects
        """
        max_results = max_results or config.ARXIV_MAX_RESULTS
        categories = categories or config.ARXIV_CATEGORIES
        
        # Check cache first
        cache_key = f"{query}_{max_results}_{'-'.join(categories)}"
        if cache_key in self.cache:
            print(f"📚 Using cached Arxiv results for: {query[:50]}...")
            return self.cache[cache_key]
        
        print(f"\n🔎 Searching Arxiv for: {query[:50]}...")
        
        # Build search query
        search_query = self._build_search_query(query, categories)
        
        try:
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers = []
            for result in self.client.results(search):
                paper = ArxivPaper(
                    arxiv_id=result.entry_id.split("/")[-1],
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    summary=result.summary,
                    published=result.published,
                    updated=result.updated,
                    categories=result.categories,
                    pdf_url=result.pdf_url,
                    arxiv_url=result.entry_id
                )
                papers.append(paper)
            
            # Cache results
            self.cache[cache_key] = papers
            
            if papers:
                print(f"✅ Found {len(papers)} relevant papers")
            else:
                print("ℹ️ No papers found for this query")
            
            return papers
            
        except Exception as e:
            print(f"❌ Arxiv search failed: {e}")
            return []
    
    def _build_search_query(self, query: str, categories: List[str]) -> str:
        """
        Build an Arxiv search query from natural language.
        
        Arxiv uses a specific query syntax:
        - ti: title
        - abs: abstract
        - au: author
        - cat: category
        - all: all fields
        """
        # Extract key terms (simple approach - could be enhanced with NLP)
        # Remove common words and keep meaningful terms
        stop_words = {
            'what', 'is', 'the', 'a', 'an', 'how', 'does', 'do', 'can', 'could',
            'would', 'should', 'will', 'about', 'for', 'in', 'on', 'with', 'to',
            'of', 'and', 'or', 'are', 'there', 'this', 'that', 'these', 'those',
            'it', 'its', "it's", 'be', 'been', 'being', 'have', 'has', 'had',
            'tell', 'me', 'explain', 'describe', 'show', 'give', 'find',
            'latest', 'recent', 'advances', 'new', 'modern', 'current'
        }
        
        words = query.lower().split()
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Use simpler "all:" query format which searches all fields
        if key_terms:
            # Join with spaces for a simpler search (arxiv handles this well)
            search_terms = " ".join(key_terms[:4])  # Limit to top 4 terms
            search_query = f"all:{search_terms}"
        else:
            search_query = f"all:{query}"
        
        # Add category filter if specified
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            search_query = f"({search_query}) AND ({cat_query})"
        
        print(f"   [Arxiv query: {search_query}]")
        return search_query
    
    def get_papers_for_query(self, question: str, answer: str) -> List[ArxivPaper]:
        """
        Get relevant papers based on a Q&A pair.
        Uses both the question and key points from the answer.
        """
        # Combine question with key answer terms
        combined_query = question
        
        # Extract key terms from answer (first 200 chars)
        answer_preview = answer[:200]
        
        return self.search_papers(combined_query)
    
    def format_papers_for_response(self, papers: List[ArxivPaper]) -> str:
        """Format papers as a readable section to append to responses."""
        if not papers:
            return ""
        
        output = "\n\n📚 **Related Research Papers:**\n"
        output += "=" * 50 + "\n"
        
        for i, paper in enumerate(papers, 1):
            output += f"\n{i}. {paper.format_for_display()}"
        
        output += "\n" + "=" * 50
        return output
    
    def get_paper_content_for_db(self, paper: ArxivPaper) -> str:
        """
        Format paper content for adding to the vector database.
        """
        return f"""Title: {paper.title}

Authors: {', '.join(paper.authors)}

Published: {paper.published.strftime('%Y-%m-%d')}

Abstract: {paper.summary}

Categories: {', '.join(paper.categories)}

Citation: {paper.format_citation()}

Source: arXiv ({paper.arxiv_url})
"""
    
    def download_paper_pdf(self, paper: ArxivPaper) -> Optional[str]:
        """
        Download a paper's PDF to the cache directory.
        Returns the file path if successful.
        """
        pdf_path = os.path.join(config.ARXIV_CACHE_PATH, f"{paper.arxiv_id}.pdf")
        
        if os.path.exists(pdf_path):
            return pdf_path
        
        try:
            print(f"📥 Downloading: {paper.title[:50]}...")
            
            # Use arxiv library to download
            search = arxiv.Search(id_list=[paper.arxiv_id])
            result = next(self.client.results(search))
            result.download_pdf(dirpath=config.ARXIV_CACHE_PATH, filename=f"{paper.arxiv_id}.pdf")
            
            return pdf_path
            
        except Exception as e:
            print(f"❌ Failed to download PDF: {e}")
            return None
    
    def should_fetch_papers(self, question: str) -> bool:
        """
        Determine if a question would benefit from Arxiv papers.
        Looks for research-related keywords.
        """
        research_indicators = [
            'research', 'paper', 'study', 'studies', 'scientific',
            'literature', 'published', 'journal', 'academic',
            'theory', 'hypothesis', 'experiment', 'findings',
            'state of the art', 'sota', 'latest', 'recent',
            'advances', 'breakthrough', 'novel', 'approach',
            'method', 'algorithm', 'model', 'framework',
            'survey', 'review', 'comparison', 'benchmark',
            'optimization', 'performance', 'efficient', 'improve',
            'machine learning', 'deep learning', 'neural', 'ai',
            'scheduling', 'distributed', 'parallel', 'concurrent'
        ]
        
        question_lower = question.lower()
        matched = [ind for ind in research_indicators if ind in question_lower]
        if matched:
            print(f"\n📚 [Arxiv: Matched keywords: {', '.join(matched)}]")
        return len(matched) > 0
    
    def clear_cache(self) -> None:
        """Clear the paper cache."""
        self.cache = {}
        print("🗑️ Arxiv cache cleared")
