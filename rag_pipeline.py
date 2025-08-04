# import os
# import tempfile
# import uuid
# import time
# import json
# from typing import List, Optional, Dict, Any
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
# from langchain_chroma import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_ollama import OllamaEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# import google.generativeai as genai
# from google.generativeai import GenerativeModel
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class EnhancedRAGPipeline:
#     def __init__(self, api_key: str = None, chroma_path: str = "chroma_db", use_huggingface: bool = False):
#         """
#         Initialize enhanced RAG pipeline with persistent storage and advanced features
        
#         Args:
#             api_key: Google API key for Gemini
#             chroma_path: Path for persistent Chroma database
#             use_huggingface: Whether to use HuggingFace embeddings instead of Ollama (default: False)
#         """
#         self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
#         if not self.api_key:
#             raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
#         genai.configure(api_key=self.api_key)
#         self.model = GenerativeModel("gemini-1.5-pro")
        
#         # Vector database configuration
#         self.chroma_path = chroma_path
#         self.use_huggingface = use_huggingface
        
#         # Choose embedding function - Default to Ollama
#         if use_huggingface:
#             try:
#                 self.embedding_function = HuggingFaceEmbeddings(
#                     model_name="sentence-transformers/all-MiniLM-L6-v2"
#                 )
#                 logger.info("Using HuggingFace embeddings (all-MiniLM-L6-v2)")
#             except Exception as e:
#                 logger.warning(f"HuggingFace not available, falling back to Ollama: {e}")
#                 self.embedding_function = OllamaEmbeddings(model="nomic-embed-text")
#         else:
#             try:
#                 self.embedding_function = OllamaEmbeddings(model="nomic-embed-text")
#                 logger.info("Using Ollama embeddings (nomic-embed-text)")
#             except Exception as e:
#                 logger.warning(f"Ollama not available, falling back to HuggingFace: {e}")
#                 self.embedding_function = HuggingFaceEmbeddings(
#                     model_name="sentence-transformers/all-MiniLM-L6-v2"
#                 )
        
#         # Text splitter for document chunking
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=800,
#             chunk_overlap=80,
#             length_function=len,
#         )
        
#         # Query classification categories
#         self.query_categories = [
#             'factual', 'definition', 'causal', 'how-to', 'comparison', 'summarization'
#         ]
    
#     def _check_database_exists_and_populated(self) -> bool:
#         """Check if database exists and has documents"""
#         try:
#             if not os.path.exists(self.chroma_path):
#                 logger.warning(f"Database directory does not exist: {self.chroma_path}")
#                 return False
            
#             # Try to load the database
#             db = Chroma(
#                 persist_directory=self.chroma_path,
#                 embedding_function=self.embedding_function
#             )
            
#             # Check if it has any documents
#             existing_items = db.get(include=[])
#             document_count = len(existing_items["ids"]) if existing_items["ids"] else 0
            
#             if document_count == 0:
#                 logger.warning("Database exists but contains no documents")
#                 return False
            
#             logger.info(f"Database found with {document_count} documents")
#             return True
            
#         except Exception as e:
#             logger.error(f"Error checking database: {e}")
#             return False

#     #ADDED DEF CLEAN NUMBERED RESPONSE
    
#     def _clean_numbered_response(self, response: str) -> str:
#     """Remove numbering from AI responses (1., 2., 3., etc.)"""
#         import re
#     # Remove patterns like "1. ", "2. ", "3. " at the beginning of the response
#         cleaned = re.sub(r'^\d+\.\s*', '', response.strip())
#     # Also remove patterns like "1)", "2)", "3)" at the beginning
#         cleaned = re.sub(r'^\d+\)\s*', '', cleaned)
#         return cleaned.strip()
    
#     # def _invoke_gemini(self, prompt: str, model: str = "gemini-1.5-pro") -> str:
#     #     """Wrapper for Gemini API calls"""
#     #     try:
#     #         response = self.model.generate_content(prompt)
#     #         return response.text.strip()
#     #     except Exception as e:
#     #         logger.error(f"Gemini API error: {e}")
#     #         raise

#     def _invoke_gemini(self, prompt: str, model: str = "gemini-1.5-pro") -> str:
#     """Wrapper for Gemini API calls"""
#         try:
#             response = self.model.generate_content(prompt)
#         # Clean numbered responses
#             cleaned_response = self._clean_numbered_response(response.text.strip())
#             return cleaned_response
#         except Exception as e:
#             logger.error(f"Gemini API error: {e}")
#             raise
    
#     def classify_query(self, query: str) -> str:
#         """Classify query into predefined categories"""
#         try:
#             prompt = (
#                 f"Classify this query into one of the following categories:\n"
#                 f"{self.query_categories}\n"
#                 f"Query: \"{query}\"\n"
#                 f"Respond with just the category label."
#             )
#             return self._invoke_gemini(prompt)
#         except Exception as e:
#             logger.warning(f"Query classification failed: {e}")
#             return "factual"  # Default category
    
#     def summarize_context(self, context_text: str) -> str:
#         """Summarize context if it's too long"""
#         try:
#             if len(context_text) <= 3000:
#                 return context_text
            
#             prompt = f"Summarize the following context concisely:\n{context_text}"
#             return self._invoke_gemini(prompt)
#         except Exception as e:
#             logger.warning(f"Context summarization failed: {e}")
#             return context_text[:3000]  # Truncate as fallback
    
#     def load_pdf_documents(self, pdf_directory: str) -> List[Document]:
#         """Load PDF documents from directory"""
#         try:
#             loader = PyPDFDirectoryLoader(pdf_directory)
#             documents = loader.load()
#             logger.info(f"Loaded {len(documents)} PDF documents")
#             return documents
#         except Exception as e:
#             logger.error(f"Error loading PDFs: {e}")
#             raise
    
#     def split_documents(self, documents: List[Document]) -> List[Document]:
#         """Split documents into chunks"""
#         try:
#             chunks = self.text_splitter.split_documents(documents)
#             logger.info(f"Split into {len(chunks)} chunks")
#             return chunks
#         except Exception as e:
#             logger.error(f"Error splitting documents: {e}")
#             raise
    
#     def create_vector_store_from_text(self, text: str, source_name: str = "text_input", show_embeddings: bool = False) -> Chroma:
#         """Create vector store from text with persistent storage and optional embedding visualization"""
#         try:
#             # Split text into chunks
#             chunks = self.text_splitter.split_text(text)
            
#             if not chunks:
#                 raise ValueError("No text chunks created from document")
            
#             # Create documents with proper metadata
#             documents = []
#             for i, chunk in enumerate(chunks):
#                 chunk_id = f"{source_name}:0:{i}"  # source:page:chunk_index
#                 doc = Document(
#                     page_content=chunk,
#                     metadata={
#                         "id": chunk_id,
#                         "chunk_id": i,
#                         "source": source_name,
#                         "page": 0
#                     }
#                 )
#                 documents.append(doc)
            
#             # Add to persistent vector store with optional embedding visualization
#             return self._add_to_chroma(documents, show_embeddings=show_embeddings)
            
#         except Exception as e:
#             logger.error(f"Error creating vector store: {str(e)}")
#             raise
    
#     def _add_to_chroma(self, chunks: List[Document], show_embeddings: bool = False) -> Chroma:
#         """Add documents to persistent Chroma database with duplicate detection and optional embedding visualization"""
#         try:
#             # Initialize or load existing database
#             db = Chroma(
#                 persist_directory=self.chroma_path,
#                 embedding_function=self.embedding_function
#             )
            
#             # Get existing document IDs
#             existing_items = db.get(include=[])
#             existing_ids = set(existing_items["ids"]) if existing_items["ids"] else set()
            
#             # Process chunks and assign IDs
#             last_page_id = None
#             current_chunk_index = 0
#             new_chunks = []
            
#             for chunk in chunks:
#                 # Generate chunk ID if not already present
#                 if "id" not in chunk.metadata:
#                     source = chunk.metadata.get("source", "unknown")
#                     page = chunk.metadata.get("page", 0)
#                     current_page_id = f"{source}:{page}"
                    
#                     if current_page_id == last_page_id:
#                         current_chunk_index += 1
#                     else:
#                         current_chunk_index = 0
                    
#                     chunk_id = f"{current_page_id}:{current_chunk_index}"
#                     chunk.metadata["id"] = chunk_id
#                     last_page_id = current_page_id
#                 else:
#                     chunk_id = chunk.metadata["id"]
                
#                 # Add only new chunks
#                 if chunk_id not in existing_ids:
#                     new_chunks.append(chunk)
            
#             # Add new chunks to database with optional embedding visualization
#             if new_chunks:
#                 if show_embeddings:
#                     # Generate and show embeddings before adding to database
#                     texts = [chunk.page_content for chunk in new_chunks]
#                     embeddings = self.embedding_function.embed_documents(texts)
                    
#                     logger.info(f"Generated embeddings for {len(texts)} chunks")
#                     print(f"\n=== EMBEDDING VISUALIZATION ===")
                    
#                     for i, (chunk, text, emb) in enumerate(zip(new_chunks, texts, embeddings)):
#                         print(f"\n--- Chunk {i+1} ---")
#                         print(f"Source: {chunk.metadata.get('source', 'unknown')}")
#                         print(f"Page: {chunk.metadata.get('page', 'unknown')}")
#                         print(f"Chunk ID: {chunk.metadata.get('id', 'unknown')}")
#                         print(f"Text:\n{text[:200]}...")  # First 200 characters
#                         print(f"Embedding dimensions: {len(emb)}")
#                         print(f"Embedding (first 5 dims): {emb[:5]}")
#                         if len(emb) > 5:
#                             print(f"Embedding (last 5 dims): {emb[-5:]}")
                
#                 # Add documents to database
#                 db.add_documents(new_chunks)
#                 logger.info(f"Added {len(new_chunks)} new chunks to vector store")
#                 print(f"✅ Added {len(new_chunks)} new chunks to Chroma database")
#             else:
#                 logger.info("No new chunks to add")
#                 print("ℹ  No new chunks to add - all documents already exist in database")
            
#             return db
            
#         except Exception as e:
#             logger.error(f"Error adding to Chroma: {e}")
#             raise
    
#     def add_pdf_documents(self, pdf_directory: str, show_embeddings: bool = False) -> Chroma:
#         """Load PDFs and add to persistent vector store with optional embedding visualization"""
#         try:
#             documents = self.load_pdf_documents(pdf_directory)
#             chunks = self.split_documents(documents)
            
#             if show_embeddings:
#                 print(f"\n=== PROCESSING {len(chunks)} CHUNKS FROM PDF DOCUMENTS ===")
#                 print("Sample chunk content:")
#                 if chunks:
#                     print(f"First chunk preview:\n{chunks[0].page_content[:300]}...")
            
#             return self._add_to_chroma(chunks, show_embeddings=show_embeddings)
#         except Exception as e:
#             logger.error(f"Error adding PDF documents: {e}")
#             raise
    
#     def generate_hypothetical_answer(self, question: str) -> str:
#         """Generate hypothetical answer using HyDE technique"""
#         try:
#             prompt = f"Generate a brief, hypothetical answer to the following question (even if not grounded):\n{question}"
#             return self._invoke_gemini(prompt)
#         except Exception as e:
#             logger.warning(f"HyDE generation failed: {str(e)}, using original question")
#             return question
    
#     def query_rag(self, query_text: str, include_metrics: bool = False, show_hypothetical: bool = False) -> Dict[str, Any]:
#         """
#         Enhanced RAG query with classification, HyDE, and context summarization
        
#         Args:
#             query_text: The question to answer
#             include_metrics: Whether to include performance metrics
#             show_hypothetical: Whether to display the hypothetical answer used for retrieval
            
#         Returns:
#             Dictionary with question, answer, and optional metrics
#         """
#         start_time = time.time()
        
#         try:
#             # CRITICAL FIX: Check if database exists and is populated
#             if not self._check_database_exists_and_populated():
#                 error_msg = (
#                     "❌ VECTOR DATABASE NOT FOUND OR EMPTY!\n"
#                     f"Database path: {self.chroma_path}\n"
#                     "Please add documents first using:\n"
#                     "- pipeline.add_pdf_documents('pdf_directory')\n"
#                     "- pipeline.create_vector_store_from_text('your_text', 'source_name')\n"
#                     "- Or run with --load-pdfs argument"
#                 )
#                 logger.error(error_msg)
#                 return {
#                     "question": query_text,
#                     "answer": error_msg,
#                     "query_type": "error",
#                     "processing_time": time.time() - start_time if include_metrics else None
#                 }
            
#             # Load existing vector store
#             db = Chroma(
#                 persist_directory=self.chroma_path,
#                 embedding_function=self.embedding_function
#             )
            
#             # Classify query
#             query_type = self.classify_query(query_text)
#             logger.info(f"Query classified as: {query_type}")
            
#             # Generate hypothetical answer for better retrieval
#             hypothetical_answer = self.generate_hypothetical_answer(query_text)
            
#             if show_hypothetical:
#                 print(f"\n=== HYDE HYPOTHETICAL ANSWER ===")
#                 print(f"Original Query: {query_text}")
#                 print(f"Query Type: {query_type}")
#                 print(f"Hypothetical Answer: {hypothetical_answer}")
#                 print("=" * 50)
            
#             # Retrieve relevant documents with scores
#             results = db.similarity_search_with_score(hypothetical_answer, k=5)
            
#             if not results:
#                 return {
#                     "question": query_text,
#                     "answer": "I couldn't find relevant information in the database to answer this question.",
#                     "query_type": query_type,
#                     "hypothetical_answer": hypothetical_answer if show_hypothetical else None,
#                     "processing_time": time.time() - start_time if include_metrics else None
#                 }
            
#             # Extract documents and format context
#             top_docs = [doc for doc, score in results]
#             context_text = "\n\n---\n\n".join([doc.page_content for doc in top_docs])
            
#             if include_metrics or show_hypothetical:
#                 print(f"\n=== RETRIEVED DOCUMENTS ===")
#                 for i, (doc, score) in enumerate(results):
#                     print(f"Doc {i+1} (Score: {score:.4f}):")
#                     print(f"Source: {doc.metadata.get('source', 'unknown')}")
#                     print(f"Preview: {doc.page_content[:150]}...")
#                     print("-" * 40)
            
#             # Summarize context if too long
#             summary = self.summarize_context(context_text)
#             was_summarized = len(summary) < len(context_text)
            
#             if was_summarized and (include_metrics or show_hypothetical):
#                 print(f"\n=== CONTEXT SUMMARIZATION ===")
#                 print(f"Original context length: {len(context_text)} chars")
#                 print(f"Summarized context length: {len(summary)} chars")
#                 print(f"Reduction: {((len(context_text) - len(summary)) / len(context_text) * 100):.1f}%")
            
#             # Create prompt template
# #             PROMPT_TEMPLATE = """
# # Answer the question based only on the following context:
# # {context}

# # ---
# # Answer the question: {question}
# # """
#             PROMPT_TEMPLATE = """
# Answer the question based only on the following context:
# {context}

# ---
# Answer the question: {question}

# Please provide a direct answer without any numbering, bullet points, or prefixes.
# """
            
#             prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#             prompt = prompt_template.format(context=summary, question=query_text)
            
#             # Generate final answer
#             response_text = self._invoke_gemini(prompt)
            
#             # Prepare response
#             response_data = {
#                 "question": query_text,
#                 "answer": response_text,
#                 "query_type": query_type
#             }
            
#             if show_hypothetical:
#                 response_data["hypothetical_answer"] = hypothetical_answer
            
#             if include_metrics:
#                 response_data.update({
#                     "processing_time": time.time() - start_time,
#                     "documents_retrieved": len(top_docs),
#                     "context_length": len(context_text),
#                     "summarized": was_summarized,
#                     "final_context_length": len(summary),
#                     "retrieval_scores": [score for _, score in results]
#                 })
            
#             return response_data
            
#         except Exception as e:
#             logger.error(f"Error in RAG query: {e}")
#             return {
#                 "question": query_text,
#                 "answer": f"Error processing question: {str(e)}",
#                 "query_type": "error",
#                 "processing_time": time.time() - start_time if include_metrics else None
#             }
    
#     def answer_question(self, question: str) -> str:
#         """Simple interface for single question answering"""
#         result = self.query_rag(question)
#         return result["answer"]
    
#     def process_questions(self, questions: List[str], include_metrics: bool = False, show_details: bool = False) -> List[Dict[str, Any]]:
#         """Process multiple questions with enhanced features and optional detailed output"""
#         try:
#             if not questions:
#                 return []
            
#             results = []
#             for i, question in enumerate(questions):
#                 if not question or not question.strip():
#                     results.append({
#                         "question": question,
#                         "answer": "Error: Question is empty",
#                         "query_type": "error"
#                     })
#                     continue
                
#                 if show_details:
#                     print(f"\n{'='*60}")
#                     print(f"PROCESSING QUESTION {i+1}/{len(questions)}")
#                     print(f"{'='*60}")
                
#                 result = self.query_rag(
#                     question.strip(), 
#                     include_metrics=include_metrics, 
#                     show_hypothetical=show_details
#                 )
#                 results.append(result)
                
#                 if show_details:
#                     print(f"\n=== FINAL ANSWER ===")
#                     print(result["answer"])
            
#             return results
            
#         except Exception as e:
#             logger.error(f"Error processing questions: {str(e)}")
#             return [{
#                 "question": q,
#                 "answer": f"Error processing questions: {str(e)}",
#                 "query_type": "error"
#             } for q in questions]
    
#     def get_database_stats(self) -> Dict[str, Any]:
#         """Get statistics about the vector database"""
#         try:
#             if not os.path.exists(self.chroma_path):
#                 return {
#                     "error": f"Database directory does not exist: {self.chroma_path}",
#                     "total_documents": 0,
#                     "database_exists": False
#                 }
            
#             db = Chroma(
#                 persist_directory=self.chroma_path,
#                 embedding_function=self.embedding_function
#             )
            
#             existing_items = db.get(include=["metadatas"])
            
#             stats = {
#                 "total_documents": len(existing_items["ids"]) if existing_items["ids"] else 0,
#                 "database_path": self.chroma_path,
#                 "database_exists": True,
#                 "embedding_model": "HuggingFace (all-MiniLM-L6-v2)" if self.use_huggingface else "Ollama (nomic-embed-text)"
#             }
            
#             # Count unique sources
#             if existing_items["metadatas"]:
#                 sources = set()
#                 for metadata in existing_items["metadatas"]:
#                     if metadata and "source" in metadata:
#                         sources.add(metadata["source"])
#                 stats["unique_sources"] = len(sources)
#                 stats["sources"] = list(sources)
#             else:
#                 stats["unique_sources"] = 0
#                 stats["sources"] = []
            
#             return stats
            
#         except Exception as e:
#             logger.error(f"Error getting database stats: {e}")
#             return {"error": str(e), "database_exists": False}

# # Global instance
# enhanced_rag_pipeline = None

# def initialize_enhanced_rag_pipeline(api_key: str, chroma_path: str = "chroma_db", use_huggingface: bool = False):
#     """Initialize the global enhanced RAG pipeline instance with Ollama as default"""
#     global enhanced_rag_pipeline
#     enhanced_rag_pipeline = EnhancedRAGPipeline(api_key, chroma_path, use_huggingface)

# def process_questions_with_enhanced_rag(questions: List[str], include_metrics: bool = False) -> List[Dict[str, Any]]:
#     """Process questions using the global enhanced RAG pipeline"""
#     if enhanced_rag_pipeline is None:
#         raise ValueError("Enhanced RAG pipeline not initialized. Call initialize_enhanced_rag_pipeline() first.")
    
#     return enhanced_rag_pipeline.process_questions(questions, include_metrics)

# # Example usage
# if __name__ == "__main__":
#     import sys
    
#     # Initialize pipeline
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         print("Please set GOOGLE_API_KEY environment variable")
#         sys.exit(1)
    
#     pipeline = EnhancedRAGPipeline(api_key=api_key, chroma_path="chroma_db", use_huggingface=False)
    
#     # Check command line arguments for different modes
#     if len(sys.argv) > 1:
#         if sys.argv[1] == "--load-pdfs":
#             # Load PDFs with embedding visualization
#             if os.path.exists("pdfs"):
#                 print("Loading PDF documents with embedding visualization...")
#                 pipeline.add_pdf_documents("pdfs", show_embeddings=True)
#             else:
#                 print("PDFs directory not found. Creating sample text document...")
#                 sample_text = """
#                 This is a sample document about artificial intelligence and machine learning.
#                 AI has many applications in various fields including healthcare, finance, and education.
#                 Machine learning is a subset of AI that focuses on algorithms that can learn from data.
#                 Natural language processing (NLP) is another important area of AI that deals with text analysis.
#                 Deep learning uses neural networks with multiple layers to solve complex problems.
#                 """
#                 pipeline.create_vector_store_from_text(sample_text, "sample_ai_doc", show_embeddings=True)
        
#         elif sys.argv[1] == "--stats":
#             # Show database statistics
#             stats = pipeline.get_database_stats()
#             print("\n=== DATABASE STATISTICS ===")
#             print(json.dumps(stats, indent=2))
        
#         else:
#             # Process query with detailed output
#             query = sys.argv[1]
#             print(f"Processing query with detailed validation: {query}")
#             result = pipeline.query_rag(query, include_metrics=True, show_hypothetical=True)
#             print(f"\n{'='*60}")
#             print("FINAL RESULT")
#             print(f"{'='*60}")
#             print(json.dumps(result, indent=2))
    
#     else:
#         print("Enhanced RAG Pipeline Usage:")
#         print("  python script.py 'your question here'    - Process a query with detailed output")
#         print("  python script.py --load-pdfs             - Load PDFs with embedding visualization")
#         print("  python script.py --stats                 - Show database statistics")
#         print("\nExample queries:")
#         print("  python script.py 'What is artificial intelligence?'")
#         print("  python script.py 'How does machine learning work?'")
        
#         # Show current database stats if available
#         try:
#             stats = pipeline.get_database_stats()
#             if stats.get("database_exists", False):
#                 if stats.get("total_documents", 0) > 0:
#                     print(f"\nCurrent database contains {stats['total_documents']} documents")
#                     if "sources" in stats and stats['sources']:
#                         print(f"Sources: {', '.join(stats['sources'])}")
#                 else:
#                     print(f"\n  Database exists but is empty at: {stats.get('database_path', 'unknown')}")
#             else:
#                 print(f"\nNo database found. Use --load-pdfs to add documents.")
#         except Exception as e:
#             print(f"\n Error checking database: {e}")

import os
import tempfile
import uuid
import time
import json
from typing import List, Optional, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
import google.generativeai as genai
from google.generativeai import GenerativeModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRAGPipeline:
    def __init__(self, api_key: str = None, chroma_path: str = "chroma_db", use_huggingface: bool = False):
        """
        Initialize enhanced RAG pipeline with persistent storage and advanced features
        
        Args:
            api_key: Google API key for Gemini
            chroma_path: Path for persistent Chroma database
            use_huggingface: Whether to use HuggingFace embeddings instead of Ollama (default: False)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        self.model = GenerativeModel("gemini-1.5-pro")
        
        # Vector database configuration
        self.chroma_path = chroma_path
        self.use_huggingface = use_huggingface
        
        # Choose embedding function - Default to Ollama
        if use_huggingface:
            try:
                self.embedding_function = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info("Using HuggingFace embeddings (all-MiniLM-L6-v2)")
            except Exception as e:
                logger.warning(f"HuggingFace not available, falling back to Ollama: {e}")
                self.embedding_function = OllamaEmbeddings(model="nomic-embed-text")
        else:
            try:
                self.embedding_function = OllamaEmbeddings(model="nomic-embed-text")
                logger.info("Using Ollama embeddings (nomic-embed-text)")
            except Exception as e:
                logger.warning(f"Ollama not available, falling back to HuggingFace: {e}")
                self.embedding_function = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
        
        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
        )
        
        # Query classification categories
        self.query_categories = [
            'factual', 'definition', 'causal', 'how-to', 'comparison', 'summarization'
        ]
    
    def _check_database_exists_and_populated(self) -> bool:
        """Check if database exists and has documents"""
        try:
            if not os.path.exists(self.chroma_path):
                logger.warning(f"Database directory does not exist: {self.chroma_path}")
                return False
            
            # Try to load the database
            db = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embedding_function
            )
            
            # Check if it has any documents
            existing_items = db.get(include=[])
            document_count = len(existing_items["ids"]) if existing_items["ids"] else 0
            
            if document_count == 0:
                logger.warning("Database exists but contains no documents")
                return False
            
            logger.info(f"Database found with {document_count} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error checking database: {e}")
            return False

    #ADDED DEF CLEAN NUMBERED RESPONSE
    
    def _clean_numbered_response(self, response: str) -> str:
        """Remove numbering from AI responses (1., 2., 3., etc.)"""
        import re
        # Remove patterns like "1. ", "2. ", "3. " at the beginning of the response
        cleaned = re.sub(r'^\d+\.\s*', '', response.strip())
        # Also remove patterns like "1)", "2)", "3)" at the beginning
        cleaned = re.sub(r'^\d+\)\s*', '', cleaned)
        return cleaned.strip()
    
    # def _invoke_gemini(self, prompt: str, model: str = "gemini-1.5-pro") -> str:
    #     """Wrapper for Gemini API calls"""
    #     try:
    #         response = self.model.generate_content(prompt)
    #         return response.text.strip()
    #     except Exception as e:
    #         logger.error(f"Gemini API error: {e}")
    #         raise

    def _invoke_gemini(self, prompt: str, model: str = "gemini-1.5-pro") -> str:
        """Wrapper for Gemini API calls"""
        try:
            response = self.model.generate_content(prompt)
            # Clean numbered responses
            cleaned_response = self._clean_numbered_response(response.text.strip())
            return cleaned_response
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def classify_query(self, query: str) -> str:
        """Classify query into predefined categories"""
        try:
            prompt = (
                f"Classify this query into one of the following categories:\n"
                f"{self.query_categories}\n"
                f"Query: \"{query}\"\n"
                f"Respond with just the category label."
            )
            return self._invoke_gemini(prompt)
        except Exception as e:
            logger.warning(f"Query classification failed: {e}")
            return "factual"  # Default category
    
    def summarize_context(self, context_text: str) -> str:
        """Summarize context if it's too long"""
        try:
            if len(context_text) <= 3000:
                return context_text
            
            prompt = f"Summarize the following context concisely:\n{context_text}"
            return self._invoke_gemini(prompt)
        except Exception as e:
            logger.warning(f"Context summarization failed: {e}")
            return context_text[:3000]  # Truncate as fallback
    
    def load_pdf_documents(self, pdf_directory: str) -> List[Document]:
        """Load PDF documents from directory"""
        try:
            loader = PyPDFDirectoryLoader(pdf_directory)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} PDF documents")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDFs: {e}")
            raise
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise
    
    def create_vector_store_from_text(self, text: str, source_name: str = "text_input", show_embeddings: bool = False) -> Chroma:
        """Create vector store from text with persistent storage and optional embedding visualization"""
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            if not chunks:
                raise ValueError("No text chunks created from document")
            
            # Create documents with proper metadata
            documents = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{source_name}:0:{i}"  # source:page:chunk_index
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "id": chunk_id,
                        "chunk_id": i,
                        "source": source_name,
                        "page": 0
                    }
                )
                documents.append(doc)
            
            # Add to persistent vector store with optional embedding visualization
            return self._add_to_chroma(documents, show_embeddings=show_embeddings)
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def _add_to_chroma(self, chunks: List[Document], show_embeddings: bool = False) -> Chroma:
        """Add documents to persistent Chroma database with duplicate detection and optional embedding visualization"""
        try:
            # Initialize or load existing database
            db = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embedding_function
            )
            
            # Get existing document IDs
            existing_items = db.get(include=[])
            existing_ids = set(existing_items["ids"]) if existing_items["ids"] else set()
            
            # Process chunks and assign IDs
            last_page_id = None
            current_chunk_index = 0
            new_chunks = []
            
            for chunk in chunks:
                # Generate chunk ID if not already present
                if "id" not in chunk.metadata:
                    source = chunk.metadata.get("source", "unknown")
                    page = chunk.metadata.get("page", 0)
                    current_page_id = f"{source}:{page}"
                    
                    if current_page_id == last_page_id:
                        current_chunk_index += 1
                    else:
                        current_chunk_index = 0
                    
                    chunk_id = f"{current_page_id}:{current_chunk_index}"
                    chunk.metadata["id"] = chunk_id
                    last_page_id = current_page_id
                else:
                    chunk_id = chunk.metadata["id"]
                
                # Add only new chunks
                if chunk_id not in existing_ids:
                    new_chunks.append(chunk)
            
            # Add new chunks to database with optional embedding visualization
            if new_chunks:
                if show_embeddings:
                    # Generate and show embeddings before adding to database
                    texts = [chunk.page_content for chunk in new_chunks]
                    embeddings = self.embedding_function.embed_documents(texts)
                    
                    logger.info(f"Generated embeddings for {len(texts)} chunks")
                    print(f"\n=== EMBEDDING VISUALIZATION ===")
                    
                    for i, (chunk, text, emb) in enumerate(zip(new_chunks, texts, embeddings)):
                        print(f"\n--- Chunk {i+1} ---")
                        print(f"Source: {chunk.metadata.get('source', 'unknown')}")
                        print(f"Page: {chunk.metadata.get('page', 'unknown')}")
                        print(f"Chunk ID: {chunk.metadata.get('id', 'unknown')}")
                        print(f"Text:\n{text[:200]}...")  # First 200 characters
                        print(f"Embedding dimensions: {len(emb)}")
                        print(f"Embedding (first 5 dims): {emb[:5]}")
                        if len(emb) > 5:
                            print(f"Embedding (last 5 dims): {emb[-5:]}")
                
                # Add documents to database
                db.add_documents(new_chunks)
                logger.info(f"Added {len(new_chunks)} new chunks to vector store")
                print(f"✅ Added {len(new_chunks)} new chunks to Chroma database")
            else:
                logger.info("No new chunks to add")
                print("ℹ  No new chunks to add - all documents already exist in database")
            
            return db
            
        except Exception as e:
            logger.error(f"Error adding to Chroma: {e}")
            raise
    
    def add_pdf_documents(self, pdf_directory: str, show_embeddings: bool = False) -> Chroma:
        """Load PDFs and add to persistent vector store with optional embedding visualization"""
        try:
            documents = self.load_pdf_documents(pdf_directory)
            chunks = self.split_documents(documents)
            
            if show_embeddings:
                print(f"\n=== PROCESSING {len(chunks)} CHUNKS FROM PDF DOCUMENTS ===")
                print("Sample chunk content:")
                if chunks:
                    print(f"First chunk preview:\n{chunks[0].page_content[:300]}...")
            
            return self._add_to_chroma(chunks, show_embeddings=show_embeddings)
        except Exception as e:
            logger.error(f"Error adding PDF documents: {e}")
            raise
    
    def generate_hypothetical_answer(self, question: str) -> str:
        """Generate hypothetical answer using HyDE technique"""
        try:
            prompt = f"Generate a brief, hypothetical answer to the following question (even if not grounded):\n{question}"
            return self._invoke_gemini(prompt)
        except Exception as e:
            logger.warning(f"HyDE generation failed: {str(e)}, using original question")
            return question
    
    def query_rag(self, query_text: str, include_metrics: bool = False, show_hypothetical: bool = False) -> Dict[str, Any]:
        """
        Enhanced RAG query with classification, HyDE, and context summarization
        
        Args:
            query_text: The question to answer
            include_metrics: Whether to include performance metrics
            show_hypothetical: Whether to display the hypothetical answer used for retrieval
            
        Returns:
            Dictionary with question, answer, and optional metrics
        """
        start_time = time.time()
        
        try:
            # CRITICAL FIX: Check if database exists and is populated
            if not self._check_database_exists_and_populated():
                error_msg = (
                    "❌ VECTOR DATABASE NOT FOUND OR EMPTY!\n"
                    f"Database path: {self.chroma_path}\n"
                    "Please add documents first using:\n"
                    "- pipeline.add_pdf_documents('pdf_directory')\n"
                    "- pipeline.create_vector_store_from_text('your_text', 'source_name')\n"
                    "- Or run with --load-pdfs argument"
                )
                logger.error(error_msg)
                return {
                    "question": query_text,
                    "answer": error_msg,
                    "query_type": "error",
                    "processing_time": time.time() - start_time if include_metrics else None
                }
            
            # Load existing vector store
            db = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embedding_function
            )
            
            # Classify query
            query_type = self.classify_query(query_text)
            logger.info(f"Query classified as: {query_type}")
            
            # Generate hypothetical answer for better retrieval
            hypothetical_answer = self.generate_hypothetical_answer(query_text)
            
            if show_hypothetical:
                print(f"\n=== HYDE HYPOTHETICAL ANSWER ===")
                print(f"Original Query: {query_text}")
                print(f"Query Type: {query_type}")
                print(f"Hypothetical Answer: {hypothetical_answer}")
                print("=" * 50)
            
            # Retrieve relevant documents with scores
            results = db.similarity_search_with_score(hypothetical_answer, k=5)
            
            if not results:
                return {
                    "question": query_text,
                    "answer": "I couldn't find relevant information in the database to answer this question.",
                    "query_type": query_type,
                    "hypothetical_answer": hypothetical_answer if show_hypothetical else None,
                    "processing_time": time.time() - start_time if include_metrics else None
                }
            
            # Extract documents and format context
            top_docs = [doc for doc, score in results]
            context_text = "\n\n---\n\n".join([doc.page_content for doc in top_docs])
            
            if include_metrics or show_hypothetical:
                print(f"\n=== RETRIEVED DOCUMENTS ===")
                for i, (doc, score) in enumerate(results):
                    print(f"Doc {i+1} (Score: {score:.4f}):")
                    print(f"Source: {doc.metadata.get('source', 'unknown')}")
                    print(f"Preview: {doc.page_content[:150]}...")
                    print("-" * 40)
            
            # Summarize context if too long
            summary = self.summarize_context(context_text)
            was_summarized = len(summary) < len(context_text)
            
            if was_summarized and (include_metrics or show_hypothetical):
                print(f"\n=== CONTEXT SUMMARIZATION ===")
                print(f"Original context length: {len(context_text)} chars")
                print(f"Summarized context length: {len(summary)} chars")
                print(f"Reduction: {((len(context_text) - len(summary)) / len(context_text) * 100):.1f}%")
            
            # Create prompt template
#             PROMPT_TEMPLATE = """
# Answer the question based only on the following context:
# {context}

# ---
# Answer the question: {question}
# """
            PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---
Answer the question: {question}

Please provide a direct answer without any numbering, bullet points, or prefixes.
"""
            
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=summary, question=query_text)
            
            # Generate final answer
            response_text = self._invoke_gemini(prompt)
            
            # Prepare response
            response_data = {
                "question": query_text,
                "answer": response_text,
                "query_type": query_type
            }
            
            if show_hypothetical:
                response_data["hypothetical_answer"] = hypothetical_answer
            
            if include_metrics:
                response_data.update({
                    "processing_time": time.time() - start_time,
                    "documents_retrieved": len(top_docs),
                    "context_length": len(context_text),
                    "summarized": was_summarized,
                    "final_context_length": len(summary),
                    "retrieval_scores": [score for _, score in results]
                })
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return {
                "question": query_text,
                "answer": f"Error processing question: {str(e)}",
                "query_type": "error",
                "processing_time": time.time() - start_time if include_metrics else None
            }
    
    def answer_question(self, question: str) -> str:
        """Simple interface for single question answering"""
        result = self.query_rag(question)
        return result["answer"]
    
    def process_questions(self, questions: List[str], include_metrics: bool = False, show_details: bool = False) -> List[Dict[str, Any]]:
        """Process multiple questions with enhanced features and optional detailed output"""
        try:
            if not questions:
                return []
            
            results = []
            for i, question in enumerate(questions):
                if not question or not question.strip():
                    results.append({
                        "question": question,
                        "answer": "Error: Question is empty",
                        "query_type": "error"
                    })
                    continue
                
                if show_details:
                    print(f"\n{'='*60}")
                    print(f"PROCESSING QUESTION {i+1}/{len(questions)}")
                    print(f"{'='*60}")
                
                result = self.query_rag(
                    question.strip(), 
                    include_metrics=include_metrics, 
                    show_hypothetical=show_details
                )
                results.append(result)
                
                if show_details:
                    print(f"\n=== FINAL ANSWER ===")
                    print(result["answer"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing questions: {str(e)}")
            return [{
                "question": q,
                "answer": f"Error processing questions: {str(e)}",
                "query_type": "error"
            } for q in questions]
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        try:
            if not os.path.exists(self.chroma_path):
                return {
                    "error": f"Database directory does not exist: {self.chroma_path}",
                    "total_documents": 0,
                    "database_exists": False
                }
            
            db = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embedding_function
            )
            
            existing_items = db.get(include=["metadatas"])
            
            stats = {
                "total_documents": len(existing_items["ids"]) if existing_items["ids"] else 0,
                "database_path": self.chroma_path,
                "database_exists": True,
                "embedding_model": "HuggingFace (all-MiniLM-L6-v2)" if self.use_huggingface else "Ollama (nomic-embed-text)"
            }
            
            # Count unique sources
            if existing_items["metadatas"]:
                sources = set()
                for metadata in existing_items["metadatas"]:
                    if metadata and "source" in metadata:
                        sources.add(metadata["source"])
                stats["unique_sources"] = len(sources)
                stats["sources"] = list(sources)
            else:
                stats["unique_sources"] = 0
                stats["sources"] = []
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e), "database_exists": False}

# Global instance
enhanced_rag_pipeline = None

def initialize_enhanced_rag_pipeline(api_key: str, chroma_path: str = "chroma_db", use_huggingface: bool = False):
    """Initialize the global enhanced RAG pipeline instance with Ollama as default"""
    global enhanced_rag_pipeline
    enhanced_rag_pipeline = EnhancedRAGPipeline(api_key, chroma_path, use_huggingface)

def process_questions_with_enhanced_rag(questions: List[str], include_metrics: bool = False) -> List[Dict[str, Any]]:
    """Process questions using the global enhanced RAG pipeline"""
    if enhanced_rag_pipeline is None:
        raise ValueError("Enhanced RAG pipeline not initialized. Call initialize_enhanced_rag_pipeline() first.")
    
    return enhanced_rag_pipeline.process_questions(questions, include_metrics)

# Example usage
if __name__ == "__main__":
    import sys
    
    # Initialize pipeline
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Please set GOOGLE_API_KEY environment variable")
        sys.exit(1)
    
    pipeline = EnhancedRAGPipeline(api_key=api_key, chroma_path="chroma_db", use_huggingface=False)
    
    # Check command line arguments for different modes
    if len(sys.argv) > 1:
        if sys.argv[1] == "--load-pdfs":
            # Load PDFs with embedding visualization
            if os.path.exists("pdfs"):
                print("Loading PDF documents with embedding visualization...")
                pipeline.add_pdf_documents("pdfs", show_embeddings=True)
            else:
                print("PDFs directory not found. Creating sample text document...")
                sample_text = """
                This is a sample document about artificial intelligence and machine learning.
                AI has many applications in various fields including healthcare, finance, and education.
                Machine learning is a subset of AI that focuses on algorithms that can learn from data.
                Natural language processing (NLP) is another important area of AI that deals with text analysis.
                Deep learning uses neural networks with multiple layers to solve complex problems.
                """
                pipeline.create_vector_store_from_text(sample_text, "sample_ai_doc", show_embeddings=True)
        
        elif sys.argv[1] == "--stats":
            # Show database statistics
            stats = pipeline.get_database_stats()
            print("\n=== DATABASE STATISTICS ===")
            print(json.dumps(stats, indent=2))
        
        else:
            # Process query with detailed output
            query = sys.argv[1]
            print(f"Processing query with detailed validation: {query}")
            result = pipeline.query_rag(query, include_metrics=True, show_hypothetical=True)
            print(f"\n{'='*60}")
            print("FINAL RESULT")
            print(f"{'='*60}")
            print(json.dumps(result, indent=2))
    
    else:
        print("Enhanced RAG Pipeline Usage:")
        print("  python script.py 'your question here'    - Process a query with detailed output")
        print("  python script.py --load-pdfs             - Load PDFs with embedding visualization")
        print("  python script.py --stats                 - Show database statistics")
        print("\nExample queries:")
        print("  python script.py 'What is artificial intelligence?'")
        print("  python script.py 'How does machine learning work?'")
        
        # Show current database stats if available
        try:
            stats = pipeline.get_database_stats()
            if stats.get("database_exists", False):
                if stats.get("total_documents", 0) > 0:
                    print(f"\nCurrent database contains {stats['total_documents']} documents")
                    if "sources" in stats and stats['sources']:
                        print(f"Sources: {', '.join(stats['sources'])}")
                else:
                    print(f"\n  Database exists but is empty at: {stats.get('database_path', 'unknown')}")
            else:
                print(f"\nNo database found. Use --load-pdfs to add documents.")
        except Exception as e:
            print(f"\n Error checking database: {e}")
