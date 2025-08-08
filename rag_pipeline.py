import os
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface'
os.environ['HF_HOME'] = '/tmp/huggingface'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import time
import json
import hashlib
import logging
from typing import List, Optional, Dict, Any
from collections import OrderedDict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
import google.generativeai as genai
from google.generativeai import GenerativeModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LightweightRAGPipeline:
    DEFAULT_CHUNK_SIZE = 1000  # Increased from 600 for better context
    DEFAULT_OVERLAP = 200      # Increased from 50 for better continuity
    DEFAULT_MAX_CACHE_SIZE = 20
    DEFAULT_BATCH_SIZE = 10
    DEFAULT_MAX_OUTPUT_TOKENS = 1024  # Increased from 512 for more detailed answers
    DEFAULT_MAX_DOCS_PDF = 10
    DEFAULT_MAX_CHUNKS_TEXT = 50      # Increased from 30
    DEFAULT_MAX_CHUNKS_SPLIT = 300    # Increased from 200

    def __init__(self, api_key: str = None, chroma_path: str = "chroma_db"):
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
                "max_output_tokens": self.DEFAULT_MAX_OUTPUT_TOKENS,
            }
        )

        self.chroma_path = chroma_path
        self.query_cache = OrderedDict()  # LRU-style cache
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=self.api_key,
            task_type="retrieval_document"
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.DEFAULT_CHUNK_SIZE,
            chunk_overlap=self.DEFAULT_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],  # Better separators for policy documents
        )

        self.db = None

    def _get_db(self):
        if self.db is None:
            try:
                self.db = Chroma(
                    embedding_function=self.embedding_function,
                    persist_directory=self.chroma_path
                )
            except Exception as e:
                logging.error("Database connection failed", exc_info=True)
                return None
        return self.db

    def _get_cache_key(self, query_text: str) -> str:
        return hashlib.md5(query_text.strip().lower().encode()).hexdigest()[:16]

    def _manage_cache(self):
        while len(self.query_cache) > self.DEFAULT_MAX_CACHE_SIZE:
            self.query_cache.popitem(last=False)  # Remove oldest

    def _clean_response(self, response: str) -> str:
        import re
        cleaned = re.sub(r'^\d+\.\s*', '', response.strip())
        cleaned = re.sub(r'^\d+\)\s*', '', cleaned)
        return cleaned.strip()

    def _invoke_gemini_fast(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return self._clean_response(response.text.strip())
        except Exception:
            logging.error("Gemini API call failed", exc_info=True)
            raise

    def load_pdf_documents(self, pdf_directory: str) -> List[Document]:
        try:
            loader = PyPDFDirectoryLoader(pdf_directory)
            documents = loader.load()
            return documents[:self.DEFAULT_MAX_DOCS_PDF]
        except Exception:
            logging.error("Failed to load PDF documents", exc_info=True)
            raise

    def split_documents(self, documents: List[Document]) -> List[Document]:
        try:
            chunks = self.text_splitter.split_documents(documents)
            return chunks[:self.DEFAULT_MAX_CHUNKS_SPLIT]
        except Exception:
            logging.error("Document splitting failed", exc_info=True)
            raise

    def create_vector_store_from_text(self, text: str, source_name: str = "text_input") -> Chroma:
        try:
            chunks = self.text_splitter.split_text(text)[:self.DEFAULT_MAX_CHUNKS_TEXT]
            if not chunks:
                raise ValueError("No text chunks created")

            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        "id": f"{source_name}:0:{i}",
                        "source": source_name,
                        "chunk_id": i
                    }
                ) for i, chunk in enumerate(chunks)
            ]
            return self._add_to_chroma(documents)
        except Exception:
            logging.error("Failed to create vector store from text", exc_info=True)
            raise

    def _add_to_chroma(self, chunks: List[Document]) -> Chroma:
        try:
            db = self._get_db()
            if db is None:
                db = Chroma(
                    embedding_function=self.embedding_function,
                    persist_directory=self.chroma_path
                )

            existing_items = db.get(include=[], limit=1000)
            existing_ids = set(existing_items["ids"] or [])

            new_chunks = [c for c in chunks if c.metadata.get("id") not in existing_ids]

            for i in range(0, len(new_chunks), self.DEFAULT_BATCH_SIZE):
                db.add_documents(new_chunks[i:i + self.DEFAULT_BATCH_SIZE])
                time.sleep(0.05)

            self.db = db
            return db
        except Exception:
            logging.error("Failed to add documents to Chroma", exc_info=True)
            raise

    def add_pdf_documents(self, pdf_directory: str) -> Chroma:
        try:
            documents = self.load_pdf_documents(pdf_directory)
            chunks = self.split_documents(documents)
            return self._add_to_chroma(chunks)
        except Exception:
            logging.error("Failed to add PDF documents", exc_info=True)
            raise

    def query_rag(self, query_text: str, include_metrics: bool = False) -> Dict[str, Any]:
        start_time = time.time()
        try:
            cache_key = self._get_cache_key(query_text)
            if cache_key in self.query_cache:
                result = self.query_cache[cache_key].copy()
                if include_metrics:
                    result.update({
                        "cache_hit": True,
                        "processing_time": time.time() - start_time
                    })
                return result

            self._manage_cache()
            db = self._get_db()
            if db is None:
                return {"question": query_text, "answer": "Database connection failed", "query_type": "error"}

            results = db.similarity_search(query_text, k=15)  # Increased from 10 for better coverage
            if not results:
                return {"question": query_text, "answer": "No relevant information found.", "query_type": "factual"}

            context_text = "\n---\n".join([doc.page_content for doc in results])[:4000] + "..."

            prompt = f"""Based on the following context from a health insurance policy document, provide a concise and accurate answer to the question. If the information is not available in the context, clearly state that.

Context:
{context_text}

Question: {query_text}

Instructions:
- Provide a brief, direct answer with key details only
- Include specific numbers, limits, or conditions if mentioned
- If information is not found, say "Not specified in the policy document"
- Keep the answer under 100 words unless more detail is essential
- Focus on the most relevant information only

Answer:"""
            answer = self._invoke_gemini_fast(prompt)

            response_data = {
                "question": query_text,
                "answer": answer,
                "query_type": "factual"
            }
            if include_metrics:
                response_data.update({
                    "processing_time": time.time() - start_time,
                    "documents_retrieved": len(results),
                    "context_length": len(context_text),
                    "cache_hit": False
                })

            self.query_cache[cache_key] = response_data.copy()
            return response_data
        except Exception as e:
            logging.error("Query failed", exc_info=True)
            return {"question": query_text, "answer": f"Error: {e}", "query_type": "error"}

    def get_database_stats(self) -> Dict[str, Any]:
        try:
            db = self._get_db()
            if db is None:
                return {"database_exists": False}
            existing_items = db.get(include=[], limit=1000)
            return {
                "total_documents": len(existing_items["ids"] or []),
                "database_exists": True,
                "cache_size": len(self.query_cache)
            }
        except Exception:
            logging.error("Failed to get DB stats", exc_info=True)
            return {"database_exists": False}

    def clear_database(self):
        self.db = None

    def clear_cache(self):
        self.query_cache.clear()

    def process_questions(self, questions: List[str], include_metrics: bool = False) -> List[Dict[str, Any]]:
        """Process multiple questions and return results"""
        results = []
        for question in questions:
            try:
                result = self.query_rag(question, include_metrics)
                results.append(result)
            except Exception as e:
                logging.error(f"Failed to process question: {question}", exc_info=True)
                results.append({
                    "question": question,
                    "answer": f"Error processing question: {str(e)}",
                    "query_type": "error"
                })
        return results


# Global instance
lightweight_rag_pipeline = None

def initialize_lightweight_rag_pipeline(api_key: str, chroma_path: str = "chroma_db"):
    global lightweight_rag_pipeline
    lightweight_rag_pipeline = LightweightRAGPipeline(api_key, chroma_path)

def process_questions_with_lightweight_rag(questions: List[str], include_metrics: bool = False) -> List[Dict[str, Any]]:
    if lightweight_rag_pipeline is None:
        raise ValueError("Pipeline not initialized.")
    return lightweight_rag_pipeline.process_questions(questions, include_metrics)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="question / --load-pdfs / --stats")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Please set GOOGLE_API_KEY environment variable")
        exit(1)

    pipeline = LightweightRAGPipeline(api_key=api_key)

    if args.command == "--load-pdfs":
        if os.path.exists("pdfs"):
            pipeline.add_pdf_documents("pdfs")
        else:
            sample_text = """
            Artificial Intelligence (AI) enables machines to perform tasks that require human intelligence.
            Machine learning is a subset of AI for predictions or decisions.
            """
            pipeline.create_vector_store_from_text(sample_text, "ai_basics")
    elif args.command == "--stats":
        print(json.dumps(pipeline.get_database_stats(), indent=2))
    else:
        result = pipeline.query_rag(args.command)
        print(result['answer'])