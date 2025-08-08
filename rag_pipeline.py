import os
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface'
os.environ['HF_HOME'] = '/tmp/huggingface'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import tempfile
import uuid
import time
import json
import hashlib
from typing import List, Optional, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
import google.generativeai as genai
from google.generativeai import GenerativeModel
import logging

# Configure logging


class LightweightRAGPipeline:
    def __init__(self, api_key: str = None, chroma_path: str = "chroma_db"):
        """Initialize lightweight RAG pipeline optimized for Render free tier"""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        
        self.model = GenerativeModel(
            "gemini-1.5-flash",
            generation_config={
                "temperature": 0.1,
                "top_p": 0.8,
                "top_k": 10,
                "max_output_tokens": 512,
            }
        )
        
        self.chroma_path = chroma_path
        self.query_cache = {}
    
        self.max_cache_size = 20
        
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=self.api_key,
            task_type="retrieval_document"
        )
       
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=50,
            length_function=len,
        )
        
        self.db = None
    
    def _get_db(self):
        """Lazy database connection"""
        if self.db is None:
            try:
                self.db = Chroma(embedding_function=self.embedding_function) 
            except Exception as e:
               
                return None
        return self.db
    
    def _get_cache_key(self, query_text: str) -> str:
        """Generate cache key"""
        return hashlib.md5(query_text.strip().lower().encode()).hexdigest()[:16]
    
    def _manage_cache(self):
        """Keep cache size limited"""
        if len(self.query_cache) >= self.max_cache_size:
            keys_to_remove = list(self.query_cache.keys())[:self.max_cache_size//2]
            for key in keys_to_remove:
                del self.query_cache[key]
    
    def _check_database_exists_and_populated(self) -> bool:
        """Quick database check"""
        try:
            db = self._get_db()
            if db is None:
                return False
            
            existing_items = db.get(limit=1, include=[])
            return len(existing_items["ids"]) > 0 if existing_items["ids"] else False
            
        except Exception:
            return False

    def _clean_response(self, response: str) -> str:
        """Clean AI response"""
        import re
        cleaned = re.sub(r'^\d+\.\s*', '', response.strip())
        cleaned = re.sub(r'^\d+\)\s*', '', cleaned)
        return cleaned.strip()

    def _invoke_gemini_fast(self, prompt: str) -> str:
        """Fast Gemini API call"""
        try:
            response = self.model.generate_content(prompt)
            return self._clean_response(response.text.strip())
        except Exception as e:
            
            raise
    
    def load_pdf_documents(self, pdf_directory: str) -> List[Document]:
        """Load PDF documents with memory limits"""
        try:
            loader = PyPDFDirectoryLoader(pdf_directory)
            documents = loader.load()
            
            if len(documents) > 10:
                documents = documents[:10]
            
            return documents
        except Exception as e:
         
            raise
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents with memory limits"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            
            if len(chunks) > 200:
                chunks = chunks[:200]
            
            return chunks
        except Exception as e:
           
            raise
    
    def create_vector_store_from_text(self, text: str, source_name: str = "text_input") -> Chroma:
        """Create vector store from text with memory limits"""
        try:
            chunks = self.text_splitter.split_text(text)
            
            if not chunks:
                raise ValueError("No text chunks created")
            
            if len(chunks) > 30:
                chunks = chunks[:30]
            
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "id": f"{source_name}:0:{i}",
                        "source": source_name,
                        "chunk_id": i
                    }
                )
                documents.append(doc)
            
            return self._add_to_chroma(documents)
            
        except Exception as e:
           
            raise
    
    def _add_to_chroma(self, chunks: List[Document]) -> Chroma:
        """Add documents to Chroma database"""
        try:
            db = Chroma(
                embedding_function=self.embedding_function
            )
            
            existing_items = db.get(include=[], limit=1000)
            existing_ids = set(existing_items["ids"]) if existing_items["ids"] else set()
            
            new_chunks = []
            for chunk in chunks:
                chunk_id = chunk.metadata.get("id")
                if chunk_id and chunk_id not in existing_ids:
                    new_chunks.append(chunk)
            
            if new_chunks:
                batch_size = 10
                for i in range(0, len(new_chunks), batch_size):
                    batch = new_chunks[i:i+batch_size]
                    db.add_documents(batch)
                    time.sleep(0.1)
                
                self.db = db
            
            return db
            
        except Exception as e:
           
            raise
    
    def add_pdf_documents(self, pdf_directory: str) -> Chroma:
        """Add PDF documents to database"""
        try:
            documents = self.load_pdf_documents(pdf_directory)
            chunks = self.split_documents(documents)
            return self._add_to_chroma(chunks)
        except Exception as e:
           
            raise
    
    def query_rag(self, query_text: str, include_metrics: bool = False) -> Dict[str, Any]:
        """Main RAG query function"""
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = self._get_cache_key(query_text)
            if cache_key in self.query_cache:
                cached_result = self.query_cache[cache_key].copy()
                if include_metrics:
                    cached_result["cache_hit"] = True
                    cached_result["processing_time"] = time.time() - start_time
                return cached_result
            
            # Manage cache size
            self._manage_cache()
            
            # Get database
            db = self._get_db()
            if db is None:
                return {
                    "question": query_text,
                    "answer": "Database connection failed",
                    "query_type": "error",
                    "processing_time": time.time() - start_time if include_metrics else None
                }
            
            # Retrieve documents
            results = db.similarity_search(query_text, k=10)
            
            if not results:
                return {
                    "question": query_text,
                    "answer": "No relevant information found in the database.",
                    "query_type": "factual",
                    "processing_time": time.time() - start_time if include_metrics else None
                }
            
            # Create context
            context_text = "\n---\n".join([doc.page_content for doc in results])
            
            if len(context_text) > 2000:
                context_text = context_text[:2000] + "..."
     
            # Generate answer
            prompt = f"""Context: {context_text}

Question: {query_text}

Answer briefly:"""
            
            response_text = self._invoke_gemini_fast(prompt)
            
            # Prepare response
            response_data = {
                "question": query_text,
                "answer": response_text,
                "query_type": "factual"
            }
            
            if include_metrics:
                response_data.update({
                    "processing_time": time.time() - start_time,
                    "documents_retrieved": len(results),
                    "context_length": len(context_text),
                    "cache_hit": False
                })
            
            # Cache result
            if cache_key:
                self.query_cache[cache_key] = response_data.copy()
            
            return response_data
            
        except Exception as e:
          
            return {
                "question": query_text,
                "answer": f"Error: {str(e)}",
                "query_type": "error",
                "processing_time": time.time() - start_time if include_metrics else None
            }
    
    def answer_question(self, question: str) -> str:
        """Simple question answering interface"""
        result = self.query_rag(question)
        return result["answer"]
    
    def process_questions(self, questions: List[str], include_metrics: bool = False) -> List[Dict[str, Any]]:
        """Process multiple questions"""
        if not questions:
            return []
        
        results = []
        for question in questions:
            if not question or not question.strip():
                results.append({
                    "question": question,
                    "answer": "Empty question",
                    "query_type": "error"
                })
                continue
            
            result = self.query_rag(question.strip(), include_metrics)
            results.append(result)
        
        return results
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            db = self._get_db()
            if db is None:
                return {"database_exists": False}
            
            existing_items = db.get(include=[], limit=1000)
            
            return {
                "total_documents": len(existing_items["ids"]) if existing_items["ids"] else 0,
                "database_exists": True,
                "cache_size": len(self.query_cache)
            }
            
        except Exception:
            return {"database_exists": False}
    
    def clear_database(self):
        """Clear database"""
        try:
            self.db = None
        except Exception as e:
            raise ValueError(f"Failed to clear database: {str(e)}")
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()

# Global instance
lightweight_rag_pipeline = None

def initialize_lightweight_rag_pipeline(api_key: str, chroma_path: str = "chroma_db"):
    """Initialize the global lightweight RAG pipeline"""
    global lightweight_rag_pipeline
    lightweight_rag_pipeline = LightweightRAGPipeline(api_key, chroma_path)

def process_questions_with_lightweight_rag(questions: List[str], include_metrics: bool = False) -> List[Dict[str, Any]]:
    """Process questions using the lightweight pipeline"""
    if lightweight_rag_pipeline is None:
        raise ValueError("Pipeline not initialized. Call initialize_lightweight_rag_pipeline() first.")
    
    return lightweight_rag_pipeline.process_questions(questions, include_metrics)

# Example usage
if __name__ == "__main__":
    import sys
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Please set GOOGLE_API_KEY environment variable")
        sys.exit(1)
    
    pipeline = LightweightRAGPipeline(api_key=api_key, chroma_path="chroma_db")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--load-pdfs":
            if os.path.exists("pdfs"):
                pipeline.add_pdf_documents("pdfs")
            else:
                sample_text = """
                Artificial Intelligence (AI) is technology that enables machines to perform tasks that typically require human intelligence.
                Machine learning is a subset of AI where algorithms learn from data to make predictions or decisions.
                Natural language processing helps computers understand and work with human language.
                Deep learning uses neural networks to solve complex problems like image recognition and language translation.
                """
                pipeline.create_vector_store_from_text(sample_text, "ai_basics")
        
        elif sys.argv[1] == "--stats":
            stats = pipeline.get_database_stats()
            print(json.dumps(stats, indent=2))
        
        else:
            query = sys.argv[1]
            result = pipeline.query_rag(query)
            print(result['answer'])
    
    else:
        print("Lightweight RAG Pipeline Usage:")
        print("  python script.py 'question'    - Ask a question")
        print("  python script.py --load-pdfs   - Load PDF documents")
        print("  python script.py --stats       - Show statistics")