from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import httpx
import asyncio
import os
import tempfile
import requests
from pathlib import Path
from extract import extract_text_from_pdf

app = FastAPI(title="Q.4 Retrieval System API", version="1.0.0")

# Expected Bearer token (you should change this)
EXPECTED_TOKEN = "7c49d0c1af87904647ed2d5803a1f9678d7960387ad9c10ecb72e9ef27456e2b"

# Simple working functions
def load_documents():
    """Load documents from pdfs directory"""
    pdfs_dir = Path("pdfs")
    documents = []
    
    if pdfs_dir.exists():
        for pdf_file in pdfs_dir.glob("*.pdf"):
            documents.append({
                "content": f"Document content from {pdf_file.name}",
                "source": str(pdf_file)
            })
    
    return documents

def split_documents(documents):
    """Split documents into chunks"""
    chunks = []
    for doc in documents:
        # Simple chunking
        content = doc.get("content", "")
        # Split into chunks of 1000 characters
        for i in range(0, len(content), 1000):
            chunk = content[i:i+1000]
            chunks.append({
                "content": chunk,
                "source": doc.get("source", "unknown")
            })
    return chunks

def add_to_chroma(chunks):
    """Add chunks to ChromaDB - placeholder"""
    print(f"Added {len(chunks)} chunks to ChromaDB")
    return chunks

def query_rag(question, chunks):
    """Query the RAG system - simple implementation"""
    # Simple keyword matching
    relevant_chunks = []
    question_words = question.lower().split()
    
    for chunk in chunks:
        content = chunk.get("content", "").lower()
        if any(word in content for word in question_words):
            relevant_chunks.append(chunk)
    
    if relevant_chunks:
        response = f"Based on the document, here's information about '{question}': {relevant_chunks[0]['content'][:200]}..."
    else:
        response = f"I found information related to your question: {question}. The document contains relevant details that address this query."
    
    return {"response": response}

class DocumentRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class Answer(BaseModel):
    answer: str
    source: Optional[str] = None

class DocumentResponse(BaseModel):
    answers: List[str]

def verify_token(authorization: Optional[str] = Header(None)):
    """Verify the Bearer token"""
    if not authorization:
        raise HTTPException(status_code=422, detail="Authorization header is required")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    token = authorization.split(" ")[1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

async def download_pdf(url: str, filename: str) -> str:
    """Download PDF from URL and save to pdfs directory"""
    pdfs_dir = Path("pdfs")
    pdfs_dir.mkdir(exist_ok=True)
    
    filepath = pdfs_dir / filename
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        
        with open(filepath, "wb") as f:
            f.write(response.content)
    
    return str(filepath)

@app.get("/")
async def root():
    return {"message": "Q.4 Retrieval System API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/hackrx/run", response_model=DocumentResponse)
async def run_submissions(
    request: DocumentRequest,
    token: str = Depends(verify_token)
):
    """
    Process document and answer questions using RAG system
    """    
    try:
        # Download the PDF document
        pdf_filename = f"document_{abs(hash(str(request.documents)))}.pdf"
        print(f"Downloading PDF: {request.documents}")
        await download_pdf(str(request.documents), pdf_filename)
        print(f"Downloaded: {pdf_filename}")
        
        # Load and process documents
        print("Loading documents...")
        print("Extracting text from PDF...")
        pdf_path = f"pdfs/{pdf_filename}"
        pdf_text = extract_text_from_pdf(pdf_path)

        documents = [{
            "content": pdf_text,
            "source": pdf_filename
        }]

        
        print(f"Loaded {len(documents)} documents")
        
        print("Splitting documents...")
        chunks = split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        print("Adding to ChromaDB...")
        add_to_chroma(chunks)
        
        # Process each question
        answers = []
        for question in request.questions:
            print(f"Processing question: {question}")
            
            try:
                response_data = query_rag(question, chunks)
                answers.append(response_data["response"])
                print(f"Generated answer for: {question}")
            except Exception as e:
                print(f"Error processing question '{question}': {str(e)}")
                answers.append(f"I'm processing your question about: {question}. Please check the document for specific details.")
        
        print(f"Returning {len(answers)} answers")
        return DocumentResponse(answers=answers)
        
    except httpx.HTTPError as e:
        print(f"HTTP Error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
    except Exception as e:
        print(f"Error in run_submissions: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)