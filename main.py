from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import httpx
import asyncio
import os
import tempfile
import requests
from pathlib import Path
from extract import query_rag_api, RAGSystem  # Import your RAG system

app = FastAPI(title="Q.4 Retrieval System API", version="1.0.0")

# Expected Bearer token (you should change this)
EXPECTED_TOKEN = "7c49d0c1af87904647ed2d5803a1f9678d7960387ad9c10ecb72e9ef27456e2b"

class DocumentRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

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
    
    async with httpx.AsyncClient(timeout=30.0) as client:  # Add timeout
        response = await client.get(url)
        response.raise_for_status()
        
        # Verify it's actually a PDF
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="URL does not point to a PDF file")
        
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
        
        # Initialize RAG system - this will automatically process the PDF
        print("Initializing RAG system...")
        rag_system = RAGSystem()
        
        # The RAG system will automatically detect the new PDF and rebuild the database
        success = rag_system.initialize_or_update_chroma_db()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to initialize RAG system")
        
        print("RAG system initialized successfully")
        
        # Process each question using the RAG system
        answers = []
        for question in request.questions:
            print(f"Processing question: {question}")
            
            try:
                # Use your sophisticated RAG system
                result = query_rag_api(question)
                
                if result.get("error"):
                    print(f"RAG system error: {result['error']}")
                    # Fallback answer
                    answers.append("I encountered an issue processing this question. Please try rephrasing it.")
                else:
                    # Extract just the answer text
                    answer = result.get("response", "").strip()
                    if answer:
                        answers.append(answer)
                    else:
                        answers.append("I couldn't find a specific answer to this question in the document.")
                
                print(f"Generated answer for: {question}")
                
            except Exception as e:
                print(f"Error processing question '{question}': {str(e)}")
                answers.append("I encountered an error while processing this question.")
        
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