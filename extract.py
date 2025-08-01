from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from transformers import pipeline
from google.generativeai import GenerativeModel
import google.generativeai as genai
import os
import time
import sys
import hashlib
import json
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = "chroma_db"
CACHE_INFO_FILE = "chroma_cache_info.json"
PDF_DIRECTORY = "pdfs"

# Set Gemini API key
genai.configure(api_key="AIzaSyBUkL0iGAl-5wcHIIozLHAIVh65aOeCyrI")  # <-- Replace with your actual key

class RAGSystem:
    def __init__(self, pdf_directory=PDF_DIRECTORY, chroma_path=CHROMA_PATH):
        self.pdf_directory = pdf_directory
        self.chroma_path = chroma_path
        self.cache_info_file = CACHE_INFO_FILE
        self.embedding_function = None
        self.db = None
        self._initialize_classifier()
        
    def _initialize_classifier(self):
        """Initialize the zero-shot classifier."""
        try:
            self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        except Exception as e:
            logger.warning(f"Could not initialize classifier: {e}")
            self.classifier = None

    def get_pdf_files_hash(self):
        """Generate a hash based on PDF files in the directory and their modification times."""
        if not os.path.exists(self.pdf_directory):
            logger.warning(f"PDF directory {self.pdf_directory} does not exist")
            return None
        
        pdf_files = []
        for file in os.listdir(self.pdf_directory):
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(self.pdf_directory, file)
                try:
                    file_stat = os.stat(file_path)
                    pdf_files.append({
                        'name': file,
                        'size': file_stat.st_size,
                        'mtime': file_stat.st_mtime
                    })
                except OSError as e:
                    logger.warning(f"Could not stat file {file_path}: {e}")
        
        if not pdf_files:
            return None
            
        # Sort for consistent hashing
        pdf_files.sort(key=lambda x: x['name'])
        
        # Create hash from file info
        hash_input = json.dumps(pdf_files, sort_keys=True)
        return hashlib.md5(hash_input.encode()).hexdigest()

    def should_rebuild_chroma_db(self):
        """Check if ChromaDB should be rebuilt based on PDF files."""
        current_hash = self.get_pdf_files_hash()
        
        if current_hash is None:
            logger.info("No PDF files found")
            return False, None
        
        if not os.path.exists(self.cache_info_file) or not os.path.exists(self.chroma_path):
            logger.info("Cache file or ChromaDB doesn't exist, will rebuild")
            return True, current_hash
        
        try:
            with open(self.cache_info_file, 'r') as f:
                cache_info = json.load(f)
            
            stored_hash = cache_info.get('pdf_hash')
            if stored_hash != current_hash:
                logger.info("PDF files have changed, will rebuild ChromaDB")
                return True, current_hash
            else:
                logger.info("PDF files unchanged, using existing ChromaDB")
                return False, current_hash
                
        except Exception as e:
            logger.warning(f"Error reading cache info: {e}, will rebuild")
            return True, current_hash

    def save_cache_info(self, pdf_hash):
        """Save cache information."""
        cache_info = {
            'pdf_hash': pdf_hash,
            'timestamp': time.time(),
            'pdf_count': len([f for f in os.listdir(self.pdf_directory) if f.lower().endswith('.pdf')])
        }
        try:
            with open(self.cache_info_file, 'w') as f:
                json.dump(cache_info, f, indent=2)
            logger.info("Cache info saved successfully")
        except Exception as e:
            logger.error(f"Error saving cache info: {e}")

    def clear_chroma_db(self):
        """Clear the existing ChromaDB."""
        try:
            if os.path.exists(self.chroma_path):
                shutil.rmtree(self.chroma_path)
                logger.info("Cleared existing ChromaDB")
        except Exception as e:
            logger.error(f"Error clearing ChromaDB: {e}")

    def load_documents(self):
        """Load PDF documents from directory."""
        try:
            if not os.path.exists(self.pdf_directory):
                logger.error(f"PDF directory {self.pdf_directory} does not exist")
                return []
                
            loader = PyPDFDirectoryLoader(self.pdf_directory)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} document pages")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []

    def split_documents(self, documents):
        """Split documents into chunks."""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=80,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Split documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            return []

    def get_embedding_function(self):
        """Get embedding function, cache it for reuse."""
        if self.embedding_function is None:
            try:
                self.embedding_function = OllamaEmbeddings(model="nomic-embed-text")
                logger.info("Embedding function initialized")
            except Exception as e:
                logger.error(f"Error initializing embedding function: {e}")
                raise
        return self.embedding_function

    def classify_query(self, query):
        """Classify the query type."""
        if self.classifier is None:
            return "factual"  # Default classification
            
        try:
            labels = ["factual", "definition", "causal", "how-to", "comparison", "summarization"]
            result = self.classifier(query, candidate_labels=labels)
            return result["labels"][0]
        except Exception as e:
            logger.warning(f"Error classifying query: {e}")
            return "factual"

    def initialize_or_update_chroma_db(self):
        """Initialize or update ChromaDB automatically."""
        try:
            should_rebuild, current_hash = self.should_rebuild_chroma_db()
            
            if current_hash is None:
                logger.warning("No PDF files found, cannot initialize database")
                return False
            
            if should_rebuild:
                logger.info("Rebuilding ChromaDB...")
                self.clear_chroma_db()
                
                # Load and process documents
                documents = self.load_documents()
                if not documents:
                    logger.error("No documents loaded, cannot build database")
                    return False
                
                chunks = self.split_documents(documents)
                if not chunks:
                    logger.error("No chunks created, cannot build database")
                    return False
                
                # Create new database
                embedding_function = self.get_embedding_function()
                self.db = Chroma(
                    persist_directory=self.chroma_path,
                    embedding_function=embedding_function
                )
                
                # Process chunks and add them
                chunk_ids = []
                processed_chunks = []
                
                last_page_id = None
                current_chunk_index = 0
                
                for chunk in chunks:
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
                    
                    processed_chunks.append(chunk)
                    chunk_ids.append(chunk_id)
                
                # Add all chunks to database
                if processed_chunks:
                    self.db.add_documents(processed_chunks, ids=chunk_ids)
                    logger.info(f"Added {len(processed_chunks)} chunks to ChromaDB")
                
                # Save cache info
                self.save_cache_info(current_hash)
                logger.info("ChromaDB rebuild completed successfully")
                
            else:
                # Initialize existing database connection
                embedding_function = self.get_embedding_function()
                self.db = Chroma(
                    persist_directory=self.chroma_path,
                    embedding_function=embedding_function
                )
                logger.info("Using existing ChromaDB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            return False

    def summarize_context(self, context_text):
        """Summarize context if it's too long."""
        try:
            model = GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(f"Summarize the following content concisely:\n{context_text}")
            return response.text
        except Exception as e:
            logger.error(f"Error summarizing context: {e}")
            return context_text  # Return original if summarization fails

    def query(self, query_text: str):
        """
        Main query method for API usage.
        Automatically handles database initialization/updates.
        """
        try:
            # Ensure database is ready
            if not self.initialize_or_update_chroma_db():
                return {
                    "error": "Failed to initialize database",
                    "response": None,
                    "sources": []
                }
            
            if self.db is None:
                return {
                    "error": "Database not available",
                    "response": None,
                    "sources": []
                }
            
            total_start = time.time()
            
            # HyDE generation
            hyde_start = time.time()
            try:
                hyde_model = GenerativeModel("gemini-1.5-pro")
                hyde_response = hyde_model.generate_content(
                    f"Answer this question briefly (hypothetical answer, even if not grounded): {query_text}"
                )
                hypothetical_answer = hyde_response.text
            except Exception as e:
                logger.warning(f"HyDE generation failed: {e}, using original query")
                hypothetical_answer = query_text
            
            hyde_end = time.time()

            # Vector retrieval
            retrieval_start = time.time()
            try:
                results = self.db.similarity_search_with_score(hypothetical_answer, k=5)
            except Exception as e:
                logger.error(f"Vector retrieval failed: {e}")
                return {
                    "error": "Vector retrieval failed",
                    "response": None,
                    "sources": []
                }
            retrieval_end = time.time()

            # Prompt preparation
            prompt_start = time.time()
            PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---
Answer the question: {question}
"""
            top_docs = [doc for doc, _ in results]
            context_text = "\n\n---\n\n".join([doc.page_content for doc in top_docs])
            
            if len(context_text) > 3000:
                summary = self.summarize_context(context_text)
            else:
                summary = context_text

            prompt = PROMPT_TEMPLATE.format(context=summary, question=query_text)
            prompt_end = time.time()

            # LLM generation
            llm_start = time.time()
            try:
                answer_model = GenerativeModel("gemini-1.5-pro")
                response = answer_model.generate_content(prompt)
                response_text = response.text
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                return {
                    "error": "Answer generation failed",
                    "response": None,
                    "sources": []
                }
            
            llm_end = time.time()
            total_end = time.time()

            sources = [doc.metadata.get("id", "unknown") for doc, _ in results]

            # Log timing information
            logger.info(f"Query processing completed in {total_end - total_start:.2f} seconds")
            
            return {
                "response": response_text,
                "sources": sources,
                "timing": {
                    "hyde_generation": hyde_end - hyde_start,
                    "vector_retrieval": retrieval_end - retrieval_start,
                    "prompt_creation": prompt_end - prompt_start,
                    "llm_generation": llm_end - llm_start,
                    "total": total_end - total_start
                },
                "query_classification": self.classify_query(query_text)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "error": str(e),
                "response": None,
                "sources": []
            }

# Global RAG system instance for API usage
rag_system = RAGSystem()

def query_rag_api(query_text: str):
    """
    API-friendly function that automatically handles everything.
    Usage: result = query_rag_api("Your question here")
    """
    return rag_system.query(query_text)

# Backward compatibility for command line usage
def query_rag(query_text: str, chunks=None):
    """Backward compatible function."""
    result = query_rag_api(query_text)
    if result.get("response"):
        print("Response:", result["response"])
        return result["response"], result.get("sources", [])
    else:
        print("Error:", result.get("error", "Unknown error"))
        return None, []

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_text = sys.argv[1]
        print(f"\nProcessing query: {query_text}")
        result = query_rag_api(query_text)
        
        if result.get("response"):
            print("\nResponse:", result["response"])
            print("\nSources:", result.get("sources", []))
            if "timing" in result:
                timing = result["timing"]
                print(f"\nTiming Info:")
                print(f"  Total: {timing['total']:.2f}s")
                print(f"  HyDE: {timing['hyde_generation']:.2f}s")
                print(f"  Retrieval: {timing['vector_retrieval']:.2f}s")
                print(f"  LLM: {timing['llm_generation']:.2f}s")
        else:
            print("Error:", result.get("error", "Unknown error"))
    else:
        print("No query provided. Usage: python extract.py 'Your question here'")
        print("\nFor API usage:")
        print("from extract import query_rag_api")
        print("result = query_rag_api('Your question here')")