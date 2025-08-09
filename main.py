from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import requests
import logging
from pypdf import PdfReader
from io import BytesIO

# Import your RAG pipeline
from rag_pipeline import LightweightRAGPipeline

app = FastAPI()

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Token for authentication
API_TOKEN = '7c49d0c1af87904647ed2d5803a1f9678d7960387ad9c10ecb72e9ef27456e2b'

security = HTTPBearer()

# Initialize RAG pipeline globally
rag_pipeline = None

def initialize_rag_pipeline():
    """Initialize the RAG pipeline with API key"""
    global rag_pipeline
    api_key = os.getenv('GOOGLE_API_KEY', 'AIzaSyCzK5gdfDGmPcQENRHdC6AhDfMh3gkwAWY')
    
    try:
        rag_pipeline = LightweightRAGPipeline(
            api_key=api_key,
            chroma_path="chroma_db"
        )
        logger.info("RAG pipeline initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
        return False

def process_pdf_with_rag(pdf_content: bytes, questions: list) -> list:
    """Process PDF content using RAG pipeline"""
    try:
        # Extract text from PDF
        pdf_file = BytesIO(pdf_content)
        reader = PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            # Clean up the text
            page_text = page_text.replace('\x00', '')  # Remove null characters
            page_text = ' '.join(page_text.split())  # Normalize whitespace
            text += page_text + "\n\n"  # Add spacing between pages
            logger.info(f"Extracted text from page {page_num + 1}, length: {len(page_text)}")
        
        logger.info(f"Total text extracted: {len(text)} characters")

        if not text.strip():
            raise ValueError("Downloaded PDF is empty or unreadable")
        
        # Log a sample of the extracted text for debugging
        logger.info(f"Sample text (first 500 chars): {text[:500]}")

        # Clear existing database and create vector store from the extracted text
        logger.info("Clearing existing database...")
        rag_pipeline.clear_database()
        
        logger.info("Creating vector store from PDF content...")
        rag_pipeline.create_vector_store_from_text(text, "current_pdf_document")
        
        # Process questions using RAG
        logger.info(f"Processing {len(questions)} questions with RAG pipeline...")
        results = rag_pipeline.process_questions(questions, include_metrics=False)
        
        # Extract just the answers
        answers = [result.get("answer", "Unable to generate answer") for result in results]
        
        return answers

    except Exception as e:
        logger.error(f"Error processing PDF with RAG: {str(e)}")
        raise

@app.post("/hackrx/run")
async def run_hackrx(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Initialize RAG pipeline if not already done
    if rag_pipeline is None:
        if not initialize_rag_pipeline():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize RAG pipeline"
            )

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON in request body")

    if "documents" not in body or not body["documents"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document URL is required")

    if "questions" not in body or not isinstance(body["questions"], list) or not body["questions"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Questions are required")

    try:
        # Download PDF
        pdf_url = body["documents"]
        logger.info(f"Downloading PDF from: {pdf_url}")
        
        # Add timeout and headers for better compatibility
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        pdf_response = requests.get(pdf_url, headers=headers, timeout=30)

        if pdf_response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"Failed to download PDF. Status code: {pdf_response.status_code}"
            )

        logger.info(f"PDF downloaded, size: {len(pdf_response.content)} bytes")

        # Process PDF using RAG pipeline instead of direct Gemini
        try:
            answers = process_pdf_with_rag(pdf_response.content, body["questions"])
            logger.info(f"Generated {len(answers)} answers using RAG pipeline")
            
        except Exception as e:
            logger.error(f"Error processing PDF with RAG: {str(e)}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to process PDF file: {str(e)}")

        # Return answers without numbering
        return JSONResponse(content={"answers": answers})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing request: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/rag-status")
async def rag_status():
    """Check RAG pipeline status and database statistics"""
    if rag_pipeline is None:
        return {"status": "not_initialized", "message": "RAG pipeline not initialized"}
    
    try:
        stats = rag_pipeline.get_database_stats()
        return {
            "status": "initialized",
            "database_stats": stats
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting RAG status: {str(e)}"
        }

# Initialize RAG pipeline on startup
@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline when the app starts"""
    logger.info("Starting up FastAPI application...")
    if not initialize_rag_pipeline():
        logger.warning("RAG pipeline initialization failed during startup")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))  # fallback for local
    uvicorn.run(app, host="0.0.0.0", port=port)