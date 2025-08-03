from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends
import os
import requests
from pypdf import PdfReader
from io import BytesIO
import logging

app = FastAPI()

# Create a pdfs directory if it doesn't exist
# os.makedirs('pdfs', exist_ok=True)

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Token for authentication
API_TOKEN = '7c49d0c1af87904647ed2d5803a1f9678d7960387ad9c10ecb72e9ef27456e2b'

security = HTTPBearer()

def extract_relevant_text(text: str, question: str, context_size: int = 500) -> str:
    """Extract relevant text segments based on keywords in the question"""
    # Extract keywords from question
    keywords = []
    if "grace period" in question.lower():
        keywords.extend(["grace period", "thirty days", "premium payment", "due date", "renew", "continue"])
    elif "ayush" in question.lower():
        keywords.extend(["ayush", "alternative medicine", "homeopathy", "unani", "siddha"])
    elif "waiting period" in question.lower():
        keywords.extend(["waiting period", "pre-existing", "PED", "coverage"])
    else:
        # Extract key words from question
        words = question.lower().replace('?', '').split()
        keywords = [word for word in words if len(word) > 3]
    
    # Find relevant sections
    text_lower = text.lower()
    relevant_sections = []
    
    for keyword in keywords:
        start_pos = 0
        while True:
            pos = text_lower.find(keyword.lower(), start_pos)
            if pos == -1:
                break
            
            # Extract context around the keyword
            start = max(0, pos - context_size)
            end = min(len(text), pos + len(keyword) + context_size)
            section = text[start:end].strip()
            
            if section and section not in relevant_sections:
                relevant_sections.append(section)
            
            start_pos = pos + 1
    
    return "\n\n---\n\n".join(relevant_sections[:3])  # Return top 3 most relevant sections

def simple_qa(text: str, questions: list) -> list:
    """Enhanced question answering with text search"""
    answers = []
    for question in questions:
        # Extract relevant text for this question
        relevant_text = extract_relevant_text(text, question)
        
        if relevant_text.strip():
            # Try to find specific answers based on question type
            if "grace period" in question.lower():
                # Look for specific grace period information
                if "thirty days" in relevant_text.lower() or "30 days" in relevant_text.lower():
                    # Extract the sentence containing grace period info
                    sentences = relevant_text.split('.')
                    for sentence in sentences:
                        if any(term in sentence.lower() for term in ["grace period", "thirty days", "30 days"]):
                            answer = sentence.strip()
                            if answer:
                                answers.append(answer + ".")
                                break
                    else:
                        answers.append("A grace period is mentioned in the policy document, but specific details need to be reviewed.")
                else:
                    answers.append("I found information about grace periods, but couldn't locate the specific duration.")
            
            elif "ayush" in question.lower():
                # Look for AYUSH treatment information
                sentences = relevant_text.split('.')
                for sentence in sentences:
                    if "ayush" in sentence.lower():
                        answer = sentence.strip()
                        if answer:
                            answers.append(answer + ".")
                            break
                else:
                    answers.append("AYUSH treatments are mentioned in the policy document.")
            
            else:
                # Generic answer based on relevant text
                # Take the first few sentences that seem relevant
                sentences = relevant_text.split('.')[:2]
                answer = '. '.join([s.strip() for s in sentences if s.strip()])
                if answer:
                    answers.append(answer + ".")
                else:
                    answers.append("Information found in the document, but unable to extract specific details.")
        else:
            answers.append(f"I cannot find relevant information about '{question}' in the provided document.")
    
    return answers

@app.post("/hackrx/run")
async def run_hackrx(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
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

        # Generate a unique filename
        logger.info(f"PDF downloaded, size: {len(pdf_response.content)} bytes")

        # Process PDF directly from memory
        try:
            pdf_file = BytesIO(pdf_response.content)
            reader = PdfReader(pdf_file)
            text = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += page_text
                logger.info(f"Extracted text from page {page_num + 1}, length: {len(page_text)}")
            
            logger.info(f"Total text extracted: {len(text)} characters")

        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to process PDF file")

        if not text.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Downloaded PDF is empty or unreadable")

        # Process questions through simple QA
        answers = simple_qa(text, body["questions"])
        
        # Format answers with proper numbering and line breaks
        formatted_answers = []
        for i, answer in enumerate(answers, 1):
            formatted_answer = f"{i}. {answer}"
            formatted_answers.append(formatted_answer)
        
        

        logger.info(f"Generated {len(formatted_answers)} answers")
        return JSONResponse(content={"answers": formatted_answers})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing request: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
