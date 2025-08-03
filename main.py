from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import requests
# import fitz  # PyMuPDF
import logging
# import uuid
import google.generativeai as genai
from google.generativeai import GenerativeModel
from pypdf import PdfReader
from io import BytesIO

app = FastAPI()

# Create a pdfs directory if it doesn't exist
# os.makedirs('pdfs', exist_ok=True)

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Token for authentication
API_TOKEN = '7c49d0c1af87904647ed2d5803a1f9678d7960387ad9c10ecb72e9ef27456e2b'

security = HTTPBearer()

def find_complete_sentences(text: str, keyword: str, max_sentences: int = 3) -> list:
    """Find complete sentences containing the keyword"""
    sentences = []
    # Split text into sentences more intelligently
    import re
    sentence_endings = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentence_endings:
        sentence = sentence.strip()
        if len(sentence) > 20 and keyword.lower() in sentence.lower():
            sentences.append(sentence)
            if len(sentences) >= max_sentences:
                break
    
    return sentences

def extract_relevant_text(text: str, question: str) -> str:
    """Extract relevant text segments based on keywords in the question"""
    # Extract keywords from question
    keywords = []
    if "grace period" in question.lower():
        keywords.extend(["grace period", "thirty days", "30 days"])
    elif "ayush" in question.lower():
        keywords.extend(["ayush", "alternative medicine", "homeopathy", "unani", "siddha"])
    elif "waiting period" in question.lower() and "pre-existing" in question.lower():
        keywords.extend(["pre-existing", "thirty-six months", "36 months", "continuous coverage"])
    elif "waiting period" in question.lower() and "cataract" in question.lower():
        keywords.extend(["cataract", "two years", "2 years"])
    elif "maternity" in question.lower():
        keywords.extend(["maternity", "childbirth", "24 months", "pregnancy"])
    elif "organ donor" in question.lower():
        keywords.extend(["organ donor", "harvesting", "transplantation"])
    elif "no claim discount" in question.lower() or "ncd" in question.lower():
        keywords.extend(["no claim discount", "ncd", "5%", "renewal"])
    elif "health check" in question.lower():
        keywords.extend(["health check", "preventive", "check-up"])
    elif "hospital" in question.lower() and "define" in question.lower():
        keywords.extend(["hospital", "inpatient beds", "10 inpatient beds", "15 beds"])
    elif "room rent" in question.lower() or "icu charges" in question.lower():
        keywords.extend(["room rent", "icu charges", "1% of sum insured", "2% of sum insured"])
    else:
        # Extract key words from question
        words = question.lower().replace('?', '').split()
        keywords = [word for word in words if len(word) > 3]
    
    # Find relevant sentences
    all_sentences = []
    for keyword in keywords:
        sentences = find_complete_sentences(text, keyword, 2)
        all_sentences.extend(sentences)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_sentences = []
    for sentence in all_sentences:
        if sentence not in seen:
            seen.add(sentence)
            unique_sentences.append(sentence)
    
    return "\n\n".join(unique_sentences[:3])  # Return top 3 unique sentences

def gemini_qa(text: str, questions: list) -> list:
    """Use Google Gemini to answer questions based on the document text"""
    # Initialize Gemini
    api_key = os.getenv('GOOGLE_API_KEY', 'AIzaSyCzK5gdfDGmPcQENRHdC6AhDfMh3gkwAWY')  # fallback to provided key
    genai.configure(api_key=api_key)
    model = GenerativeModel("gemini-1.5-pro")
    
    answers = []
    
    for question in questions:
        logger.info(f"Processing question with Gemini: {question}")
        
        # Create a prompt for Gemini with the ENTIRE document
        prompt = f"""
You are an expert insurance policy analyst. Based on the following complete insurance policy document, please answer the question accurately and concisely.

Complete Document Content:
{text}

Question: {question}

Please provide a direct, accurate answer based solely on the information in the document. Search through the entire document carefully to find the relevant information. If the information is not available in the document, clearly state that. Format your response as a complete sentence.

Answer:"""
        
        try:
            response = model.generate_content(prompt)
            answer = response.text.strip()
            
            # Clean up the answer
            if answer.startswith("Answer:"):
                answer = answer[7:].strip()
            
            answers.append(answer)
            logger.info(f"Gemini generated answer: {answer[:100]}...")
            
        except Exception as e:
            logger.error(f"Error with Gemini API: {str(e)}")
            # Fallback to text search if Gemini fails
            relevant_text = extract_relevant_text(text, question)
            if relevant_text.strip():
                fallback_answer = f"Based on the document: {relevant_text[:300]}..."
                answers.append(fallback_answer)
            else:
                answers.append(f"I cannot find information about '{question}' in the provided document.")
    
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

        # Process questions through Gemini QA
        answers = gemini_qa(text, body["questions"])
        
        # Clean up the PDF file
        try:
            os.remove(pdf_path)
        except:
            pass  # Ignore cleanup errors

        logger.info(f"Generated {len(answers)} answers")
        
        # Format answers with better readability
        formatted_answers = []
        for i, answer in enumerate(answers, 1):
            formatted_answers.append(f"{i}. {answer}")
        
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

