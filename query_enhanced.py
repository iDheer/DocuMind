"""
DocuMind Enhanced Query System
==============================
Advanced RAG system with two operation modes:

🔒 LOCAL MODE (Privacy-focused):
   - No internet connectivity required
   - Uses only local LM Studio LLM
   - Chat history still works locally
   - Perfect for sensitive documents

🧠 SELF-LEARNING MODE (Full features):
   - Gemini-based fact verification every 5 queries
   - Arxiv research paper integration
   - Dynamic database updates with new knowledge
   - Continuous improvement of knowledge base

This is the main entry point for the enhanced DocuMind system.
"""

import os
import chromadb
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# Import our new modules
import config
from chat_history import ChatHistory
from gemini_verifier import GeminiFactVerifier
from arxiv_fetcher import ArxivFetcher
from db_updater import VectorDBUpdater
from cache_manager import get_cache_manager, CacheManager
from analytics import QueryAnalytics, FeedbackCollector


class DocuMindEnhanced:
    """
    Enhanced DocuMind system with memory, verification, and research integration.
    Supports two modes: LOCAL (private) and SELF-LEARNING (online features).
    """
    
    def __init__(self, session_id: str = None, mode: str = None):
        """
        Initialize the enhanced DocuMind system.
        
        Args:
            session_id: Optional session ID to resume a previous conversation
            mode: Optional mode override ("local" or "self-learning")
        """
        # Set mode if provided
        if mode:
            config.set_mode(mode)
        
        print("=" * 60)
        print("🧠 DocuMind Enhanced - Initializing...")
        print(f"   {config.get_mode_description()}")
        print("=" * 60)
        
        # Initialize components
        self._setup_models()
        self._load_index()
        self._setup_query_engine()
        
        # Initialize enhancement modules
        self.chat_history = ChatHistory(session_id=session_id)
        self.fact_verifier = GeminiFactVerifier()
        self.arxiv_fetcher = ArxivFetcher()
        self.db_updater = VectorDBUpdater(self.storage_context)
        self.cache_manager = get_cache_manager()
        self.analytics = QueryAnalytics()
        self.feedback_collector = FeedbackCollector(self.analytics)
        
        # Feature flags based on mode
        self._apply_mode_settings()
        self.enable_caching = True  # Query result caching
        self.collect_feedback = False  # Ask for feedback after responses
        
        print("\n✅ DocuMind Enhanced ready!")
        print(f"   {config.get_mode_description()}")
        print(f"📂 Session: {self.chat_history.session_id}")
        print(f"💬 Messages in history: {len(self.chat_history)}")
        print(f"🔢 Queries this session: {self.chat_history.query_count}")
        self._print_commands_help()
    
    def _apply_mode_settings(self):
        """Apply settings based on current operation mode."""
        if config.is_local_mode():
            # Local mode: disable all online features
            self.enable_arxiv = False
            self.enable_verification = False
            self.enable_context = True  # Local chat history still works
            print("\n🔒 Running in LOCAL MODE:")
            print("   • Gemini verification: DISABLED")
            print("   • Arxiv integration: DISABLED")
            print("   • Chat context: ENABLED (local)")
        else:
            # Self-learning mode: enable all features
            self.enable_arxiv = True
            self.enable_verification = True
            self.enable_context = True
            print("\n🧠 Running in SELF-LEARNING MODE:")
            print("   • Gemini verification: ENABLED (every 5 queries)")
            print("   • Arxiv integration: ENABLED")
            print("   • Chat context: ENABLED")
            print("   • Auto DB updates: ENABLED")
    
    def _setup_models(self):
        """Configure LLM and embedding models."""
        print("\n--- Configuring models ---")
        Settings.llm = OpenAILike(
            model=config.LM_STUDIO_MODEL,
            api_base=config.LM_STUDIO_BASE_URL,
            api_key="lm-studio",  # LM Studio doesn't need a real key
            timeout=config.LM_STUDIO_TIMEOUT,
            is_chat_model=True
        )
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=config.EMBEDDING_MODEL,
            device=config.EMBEDDING_DEVICE
        )
    
    def _load_index(self):
        """Load the persisted vector index."""
        print(f"\n--- Loading data from '{config.DB_PATH}' ---")
        
        try:
            # Connect to ChromaDB
            db = chromadb.PersistentClient(path=config.DB_PATH)
            chroma_collection = db.get_or_create_collection(config.COLLECTION_NAME)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            print("--- Connected to ChromaDB ---")

            # Load storage context
            self.storage_context = StorageContext.from_defaults(
                persist_dir=config.DB_PATH, 
                vector_store=vector_store
            )
            print("--- Loaded docstore and index_store ---")
            
            # Load index
            self.index = load_index_from_storage(self.storage_context)
            print("--- Index loaded successfully ---")

        except FileNotFoundError:
            print(f"❌ Error: Storage not found at '{config.DB_PATH}'")
            print("Please run '1_build_database_advanced.py' first.")
            exit()
    
    def _setup_query_engine(self):
        """Configure retriever and query engine."""
        print("\n--- Configuring retriever with AutoMerging and Re-ranking ---")
        
        base_retriever = self.index.as_retriever(
            similarity_top_k=config.SIMILARITY_TOP_K
        )
        
        self.retriever = AutoMergingRetriever(
            base_retriever,
            self.index.storage_context,
            verbose=False  # Set to True for debugging
        )
        
        self.reranker = SentenceTransformerRerank(
            top_n=config.RERANKER_TOP_N,
            model=config.RERANKER_MODEL,
            device=config.RERANKER_DEVICE
        )
        
        # Custom QA prompt for more natural responses
        qa_prompt_template = PromptTemplate(
            """\
You are DocuMind, an intelligent document assistant. Use the following document excerpts to answer the user's question.

DOCUMENT EXCERPTS:
---------------------
{context_str}
---------------------

USER QUESTION: {query_str}

INSTRUCTIONS:
- Provide a clear, comprehensive answer based on the documents above
- You may reference concepts from the documents naturally (e.g., "According to the textbook..." or "The document explains that...")
- If the documents contain relevant information, use it to give a detailed answer
- If the documents don't fully cover the topic, say so and provide what information is available
- Be conversational and helpful, not robotic
- Include specific details, examples, or definitions when relevant

ANSWER:"""
        )
        
        self.query_engine = RetrieverQueryEngine.from_args(
            self.retriever,
            node_postprocessors=[self.reranker],
            streaming=True
        )
        
        # Update the query engine's prompt
        self.query_engine.update_prompts({
            "response_synthesizer:text_qa_template": qa_prompt_template
        })
    
    def _print_commands_help(self):
        """Print available commands."""
        print("\n📋 Available Commands:")
        print("   /help      - Show this help")
        print("   /mode      - Switch between local/self-learning modes")
        print("   /history   - Show chat history")
        print("   /clear     - Clear chat history")
        print("   /cache     - Show cache statistics / clear cache")
        print("   /verify    - Force verification now (self-learning mode)")
        print("   /arxiv     - Toggle Arxiv integration (self-learning mode)")
        print("   /sessions  - List all sessions")
        print("   /stats     - Show system statistics")
        print("   /export    - Export current session")
        print("   /analytics - View detailed analytics dashboard")
        print("   /feedback  - Toggle feedback collection on/off")
        print("   /rate      - Rate the last response (1-5)")
        print("   exit       - Exit the program")
        print()
    
    def _build_contextual_prompt(self, question: str) -> str:
        """
        Build a prompt that includes conversation context.
        """
        if not self.enable_context:
            return question
        
        context = self.chat_history.get_context_for_query()
        
        if not context:
            return question
        
        return f"""Based on our conversation history, please answer the following question.

{context}

Current Question: {question}

Please provide a helpful answer, taking into account any relevant context from our previous conversation."""
    
    def _handle_streaming_response(self, response_stream) -> str:
        """Handle streaming response with Qwen3 thinking tags."""
        full_text = ""
        in_think_tag = False
        buffer = ""
        
        print("🤖 Response: ", end="", flush=True)
        
        for text in response_stream.response_gen:
            full_text += text
            buffer += text
            
            while buffer:
                if "<think>" in buffer and not in_think_tag:
                    think_start = buffer.find("<think>")
                    if think_start > 0:
                        print(buffer[:think_start], end="", flush=True)
                    in_think_tag = True
                    print("\n\n🤔 Thinking: ", end="", flush=True)
                    buffer = buffer[think_start + 7:]
                    continue
                
                elif "</think>" in buffer and in_think_tag:
                    think_end = buffer.find("</think>")
                    if think_end > 0:
                        print(buffer[:think_end], end="", flush=True)
                    in_think_tag = False
                    print("\n\n💭 Final Answer: ", end="", flush=True)
                    buffer = buffer[think_end + 8:]
                    continue
                
                else:
                    if "<think>" not in buffer and "</think>" not in buffer:
                        print(buffer, end="", flush=True)
                        buffer = ""
                    else:
                        break
        
        if buffer:
            print(buffer, end="", flush=True)
        
        print()
        return full_text
    
    def _extract_clean_response(self, full_text: str) -> str:
        """Extract clean response text without thinking tags."""
        if "<think>" in full_text and "</think>" in full_text:
            think_end = full_text.find("</think>")
            return full_text[think_end + 8:].strip()
        return full_text
    
    def _print_sources(self, response):
        """Print source nodes from the response."""
        print("\n--- Source Nodes (Post-Reranking) ---")
        for i, node in enumerate(response.source_nodes):
            print(f"Source {i+1} (Score: {node.score:.4f}):")
            cleaned_text = ' '.join(node.get_content().split())
            print(f"  -> File: {node.metadata.get('file_name', 'N/A')}")
            print(f"  -> Content: \"{cleaned_text[:200]}...\"\n")
        print("-" * 40)
    
    def _maybe_fetch_arxiv(self, question: str, answer: str):
        """Fetch and display relevant Arxiv papers if appropriate."""
        if not self.enable_arxiv:
            return []
        
        if config.is_local_mode():
            return []
        
        # Check if this question would benefit from papers
        should_fetch = self.arxiv_fetcher.should_fetch_papers(question)
        if not should_fetch:
            print("\n📚 [Arxiv: Query doesn't match research keywords - skipping]")
            return []
        
        try:
            papers = self.arxiv_fetcher.get_papers_for_query(question, answer)
            
            if papers:
                print(self.arxiv_fetcher.format_papers_for_response(papers))
                
                # Offer to add papers to database
                if config.ARXIV_ADD_TO_DB:
                    add_response = input("\n📚 Add these papers to the knowledge base? (y/n): ").lower().strip()
                    if add_response == 'y':
                        self.db_updater.add_arxiv_papers(papers, auto_apply=True)
            else:
                print("\n📚 [Arxiv: No papers found for this query]")
            
            return papers if papers else []
        except Exception as e:
            print(f"\n⚠️ Arxiv fetch error: {e}")
            return []
    
    def _maybe_verify(self):
        """Run Gemini verification if it's time."""
        if not self.enable_verification or config.is_local_mode():
            return False
        
        if not self.chat_history.should_verify():
            return False
        
        if not config.GEMINI_API_KEY:
            print("\n⚠️ Gemini API key not set. Skipping verification.")
            return False
        
        print(f"\n🔍 Running periodic fact verification (every {config.VERIFICATION_QUERY_INTERVAL} queries)...")
        
        # Get recent Q&A pairs
        qa_pairs = self.chat_history.get_recent_qa_pairs(config.VERIFICATION_QUERY_INTERVAL)
        
        if qa_pairs:
            results = self.fact_verifier.verify_qa_pairs(qa_pairs)
            
            # Check for needed updates
            updates_needed = [r for r in results if r.needs_db_update]
            
            if updates_needed:
                print(f"\n⚠️ {len(updates_needed)} potential update(s) detected.")
                apply_response = input("Apply recommended updates? (y/n): ").lower().strip()
                if apply_response == 'y':
                    self.db_updater.apply_verification_updates(results, auto_apply=True)
            
            # Save verification log
            self.fact_verifier.save_verification_log()
            return True
        
        return False
    
    def _handle_command(self, command: str) -> bool:
        """
        Handle special commands.
        
        Returns:
            True if command was handled, False if it's a regular query
        """
        cmd = command.lower().strip()
        
        if cmd == "/help":
            self._print_commands_help()
            return True
        
        elif cmd == "/mode":
            self._handle_mode_switch()
            return True
        
        elif cmd == "/history":
            context = self.chat_history.get_context_for_query()
            if context:
                print("\n📜 Conversation History:")
                print("-" * 40)
                print(context)
                print("-" * 40)
            else:
                print("ℹ️ No history yet.")
            return True
        
        elif cmd == "/clear":
            self.chat_history.clear_history()
            return True
        
        elif cmd == "/verify":
            if config.is_local_mode():
                print("⚠️ Verification is disabled in LOCAL mode. Switch to self-learning mode with /mode")
            elif not config.GEMINI_API_KEY:
                print("⚠️ Set GEMINI_API_KEY in .env to use verification")
            else:
                qa_pairs = self.chat_history.get_recent_qa_pairs(5)
                if qa_pairs:
                    results = self.fact_verifier.verify_qa_pairs(qa_pairs)
                    self.fact_verifier.save_verification_log()
                else:
                    print("ℹ️ Not enough Q&A pairs for verification")
            return True
        
        elif cmd == "/arxiv":
            if config.is_local_mode():
                print("⚠️ Arxiv is disabled in LOCAL mode. Switch to self-learning mode with /mode")
            else:
                self.enable_arxiv = not self.enable_arxiv
                status = "enabled" if self.enable_arxiv else "disabled"
                print(f"📚 Arxiv integration {status}")
            return True
        
        elif cmd == "/sessions":
            sessions = self.chat_history.get_all_sessions()
            print("\n📂 Available Sessions:")
            for s in sessions[:10]:  # Show last 10
                marker = "👉" if s == self.chat_history.session_id else "  "
                print(f"  {marker} {s}")
            return True
        
        elif cmd == "/stats":
            print("\n📊 System Statistics:")
            print(f"   {config.get_mode_description()}")
            print(f"   Current session: {self.chat_history.session_id}")
            print(f"   Messages: {len(self.chat_history)}")
            print(f"   Queries: {self.chat_history.query_count}")
            print(f"   Arxiv: {'enabled' if self.enable_arxiv else 'disabled'}")
            print(f"   Verification: {'enabled' if self.enable_verification else 'disabled'}")
            print(f"   Caching: {'enabled' if self.enable_caching else 'disabled'}")
            self.db_updater.print_update_summary()
            self.cache_manager.print_stats()
            return True
        
        elif cmd == "/cache":
            self._handle_cache_command()
            return True
        
        elif cmd == "/export":
            self._handle_export_session()
            return True
        
        elif cmd == "/analytics":
            self.analytics.print_dashboard()
            return True
        
        elif cmd == "/feedback":
            self.collect_feedback = not self.collect_feedback
            status = "enabled" if self.collect_feedback else "disabled"
            print(f"💭 Feedback collection {status}")
            return True
        
        elif cmd == "/rate":
            self.feedback_collector.collect_feedback()
            return True
        
        return False
    
    def _handle_cache_command(self):
        """Handle cache management commands."""
        print("\n💾 Cache Management:")
        self.cache_manager.print_stats()
        
        print("\nOptions:")
        print("   1. Toggle caching on/off")
        print("   2. Clear all caches")
        print("   3. Cleanup expired entries")
        print("   4. Back to chat")
        
        choice = input("\nChoice (1-4): ").strip()
        
        if choice == "1":
            self.enable_caching = not self.enable_caching
            status = "enabled" if self.enable_caching else "disabled"
            print(f"✅ Caching {status}")
        elif choice == "2":
            confirm = input("⚠️ Clear ALL caches? (y/n): ").lower().strip()
            if confirm == 'y':
                results = self.cache_manager.clear_all()
                print(f"🗑️ Cleared: {sum(results.values())} entries")
        elif choice == "3":
            results = self.cache_manager.cleanup_all()
            print(f"🧹 Cleaned up: {sum(results.values())} expired entries")
    
    def _handle_export_session(self):
        """Export current session to a readable format."""
        import json
        from datetime import datetime
        
        export_data = {
            "session_id": self.chat_history.session_id,
            "exported_at": datetime.now().isoformat(),
            "mode": config.OPERATION_MODE,
            "summary": self.chat_history.summary,
            "messages": []
        }
        
        for msg in self.chat_history.messages:
            export_data["messages"].append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            })
        
        export_path = os.path.join(
            config.CHAT_HISTORY_PATH, 
            f"export_{self.chat_history.session_id}.json"
        )
        
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📤 Session exported to: {export_path}")
    
    def _handle_mode_switch(self):
        """Handle switching between local and self-learning modes."""
        current_mode = "local" if config.is_local_mode() else "self-learning"
        
        print(f"\n🔄 Current Mode: {config.get_mode_description()}")
        print("\nAvailable modes:")
        print("   1. 🔒 local        - Privacy-focused, no online features")
        print("   2. 🧠 self-learning - Full features with Gemini + Arxiv")
        
        choice = input("\nEnter mode (local/self-learning) or number (1/2): ").lower().strip()
        
        if choice in ["1", "local"]:
            new_mode = "local"
        elif choice in ["2", "self-learning"]:
            new_mode = "self-learning"
        else:
            print("❌ Invalid choice. Mode unchanged.")
            return
        
        if new_mode == current_mode:
            print(f"ℹ️ Already in {new_mode} mode.")
            return
        
        config.set_mode(new_mode)
        self._apply_mode_settings()
        print(f"\n✅ Switched to {config.get_mode_description()}")
    
    def query(self, question: str) -> str:
        """
        Process a user query with all enhancements.
        
        Args:
            question: The user's question
            
        Returns:
            The assistant's response
        """
        # Check for commands
        if question.startswith("/"):
            self._handle_command(question)
            return ""
        
        # Add to chat history
        self.chat_history.add_user_message(question)
        
        # Check semantic cache first (if caching enabled)
        if self.enable_caching:
            cached_result = self.cache_manager.query_cache.find_similar(question)
            if cached_result:
                cached_query, cached_data, similarity = cached_result
                print(f"\n💾 [Cache Hit - {similarity:.1%} similar to previous question]")
                clean_response = cached_data["response"]
                sources = cached_data.get("sources", [])
                
                # Add cached response to history
                self.chat_history.add_assistant_message(clean_response, sources)
                
                print(f"\n📚 Assistant: {clean_response}")
                
                if sources:
                    print("\n" + "-" * 40)
                    print("📄 Sources (from cache):")
                    for i, src in enumerate(sources[:3], 1):
                        score = src.get('score', 0)
                        print(f"   {i}. {src.get('file', 'N/A')} (score: {score:.3f})")
                
                # Record analytics (cache hit)
                self.analytics.record_query(
                    query=question,
                    response=clean_response,
                    sources=sources,
                    session_id=self.chat_history.session_id,
                    cache_hit=True
                )
                
                # Maybe collect feedback
                if self.collect_feedback:
                    self.feedback_collector.collect_feedback()
                
                # Still track for verification even with cache hits
                self._maybe_verify()
                return clean_response
        
        # Build contextual prompt
        contextual_prompt = self._build_contextual_prompt(question)
        
        # Query the engine
        response = self.query_engine.query(contextual_prompt)
        
        # Handle streaming response
        full_text = self._handle_streaming_response(response)
        clean_response = self._extract_clean_response(full_text)
        
        # Extract sources for history
        sources = []
        for node in response.source_nodes:
            sources.append({
                "file": node.metadata.get("file_name", "N/A"),
                "score": node.score,
                "snippet": node.get_content()[:200]
            })
        
        # Cache the response (if caching enabled)
        if self.enable_caching:
            self.cache_manager.query_cache.cache_query(
                question, 
                {"response": clean_response, "sources": sources}
            )
        
        # Add assistant response to history
        self.chat_history.add_assistant_message(clean_response, sources)
        
        # Print sources
        self._print_sources(response)
        
        # Track if arxiv was fetched
        arxiv_count = 0
        if self.enable_arxiv:
            papers = self._maybe_fetch_arxiv(question, clean_response)
            arxiv_count = len(papers) if papers else 0
        
        # Maybe run verification
        verification_ran = self._maybe_verify()
        
        # Record analytics
        self.analytics.record_query(
            query=question,
            response=clean_response,
            sources=sources,
            session_id=self.chat_history.session_id,
            cache_hit=False,
            arxiv_papers_count=arxiv_count,
            verification_triggered=bool(verification_ran)
        )
        
        # Maybe collect feedback
        if self.collect_feedback:
            self.feedback_collector.collect_feedback()
        
        return clean_response
    
    def run(self):
        """Run the interactive query loop."""
        print("\n" + "=" * 60)
        print(f"💬 Ready to Query! {config.get_mode_description()}")
        print("=" * 60)
        
        try:
            while True:
                question = input("\n❓ Question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() == 'exit':
                    print("\n👋 Goodbye! Session saved.")
                    break
                
                self.query(question)
        
        except KeyboardInterrupt:
            print("\n\n👋 Exiting gracefully. Session saved.")


def main():
    """Main entry point."""
    import sys
    
    session_id = None
    mode = None
    
    # Parse command line arguments
    for arg in sys.argv[1:]:
        if arg.startswith("--mode="):
            mode = arg.split("=")[1]
        elif arg in ["--local", "-l"]:
            mode = "local"
        elif arg in ["--learn", "--self-learning", "-s"]:
            mode = "self-learning"
        elif not arg.startswith("-"):
            session_id = arg
    
    # If no mode specified, ask user
    if mode is None:
        print("\n🧠 DocuMind Enhanced")
        print("=" * 40)
        print("\nSelect operation mode:")
        print("   1. 🔒 LOCAL MODE")
        print("      • Privacy-focused, completely offline")
        print("      • No data sent to external services")
        print("      • Uses only local LM Studio LLM")
        print()
        print("   2. 🧠 SELF-LEARNING MODE")
        print("      • Gemini fact verification every 5 queries")
        print("      • Arxiv research paper integration")
        print("      • Automatic knowledge base updates")
        print()
        
        choice = input("Enter choice (1/2) or mode name: ").lower().strip()
        
        if choice in ["1", "local", "l"]:
            mode = "local"
        else:
            mode = "self-learning"
    
    if session_id:
        print(f"📂 Resuming session: {session_id}")
    
    # Create and run the enhanced system
    documind = DocuMindEnhanced(session_id=session_id, mode=mode)
    documind.run()


if __name__ == "__main__":
    main()
