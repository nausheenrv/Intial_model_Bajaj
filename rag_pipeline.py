import os
import tempfile
import uuid
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai
from google.generativeai import GenerativeModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, api_key: str = None):
        """Initialize RAG pipeline with Google Gemini API key"""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        self.model = GenerativeModel("gemini-1.5-pro")
        
        # Use HuggingFace embeddings for better deployment compatibility
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def create_vector_store(self, text: str) -> Chroma:
        """Create a vector store from document text"""
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            if not chunks:
                raise ValueError("No text chunks created from document")
            
            # Create documents
            documents = [
                Document(
                    page_content=chunk,
                    metadata={"chunk_id": i, "source": "uploaded_document"}
                )
                for i, chunk in enumerate(chunks)
            ]
            
            # Create temporary directory for this session
            temp_dir = tempfile.mkdtemp()
            chroma_path = os.path.join(temp_dir, f"chroma_db_{uuid.uuid4()}")
            
            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                persist_directory=chroma_path
            )
            
            logger.info(f"Created vector store with {len(documents)} chunks")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def generate_hypothetical_answer(self, question: str) -> str:
        """Generate hypothetical answer using HyDE technique"""
        try:
            hyde_prompt = f"Answer this question briefly (hypothetical answer, even if not grounded): {question}"
            response = self.model.generate_content(hyde_prompt)
            return response.text
        except Exception as e:
            logger.warning(f"HyDE generation failed: {str(e)}, using original question")
            return question
    
    def answer_question(self, vectorstore: Chroma, question: str) -> str:
        """Answer a single question using RAG"""
        try:
            # Generate hypothetical answer for better retrieval
            hypothetical_answer = self.generate_hypothetical_answer(question)
            
            # Retrieve relevant documents
            docs = vectorstore.similarity_search(hypothetical_answer, k=5)
            
            if not docs:
                return "I couldn't find relevant information in the document to answer this question."
            
            # Prepare context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create prompt template
            prompt_template = """
Based only on the following context, answer the question. If you cannot find the answer in the context, say "I cannot find this information in the provided document."

Context:
{context}

Question: {question}

Answer:"""
            
            prompt = prompt_template.format(context=context, question=question)
            
            # Generate answer
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error answering question '{question}': {str(e)}")
            return f"Error processing question: {str(e)}"
    
    def process_questions(self, text: str, questions: List[str]) -> List[str]:
        """Process multiple questions against a document"""
        try:
            if not text or not text.strip():
                return ["Error: Document text is empty or invalid"] * len(questions)
            
            if not questions:
                return []
            
            # Create vector store
            vectorstore = self.create_vector_store(text)
            
            # Process each question
            answers = []
            for question in questions:
                if not question or not question.strip():
                    answers.append("Error: Question is empty")
                    continue
                
                answer = self.answer_question(vectorstore, question.strip())
                answers.append(answer)
            
            return answers
            
        except Exception as e:
            logger.error(f"Error processing questions: {str(e)}")
            return [f"Error processing questions: {str(e)}"] * len(questions)

# Global instance (will be initialized with API key)
rag_pipeline = None

def initialize_rag_pipeline(api_key: str):
    """Initialize the global RAG pipeline instance"""
    global rag_pipeline
    rag_pipeline = RAGPipeline(api_key)

def process_questions_with_rag(text: str, questions: List[str]) -> List[str]:
    """Process questions using the global RAG pipeline"""
    if rag_pipeline is None:
        raise ValueError("RAG pipeline not initialized. Call initialize_rag_pipeline() first.")
    
    return rag_pipeline.process_questions(text, questions)
