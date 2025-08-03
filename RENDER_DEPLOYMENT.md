# Render.com Deployment Checklist ✅

## Files Ready for Deployment
- ✅ `main.py` - FastAPI application
- ✅ `requirements.txt` - All dependencies (torch, transformers, etc.)
- ✅ `runtime.txt` - Python 3.11.9
- ✅ `render.yaml` - Render configuration
- ✅ `Procfile` - Alternative startup command
- ✅ `.gitignore` - Clean deployment

## Key Dependencies Verified
- ✅ FastAPI + Uvicorn (web server)
- ✅ PyMuPDF 1.23.14 (PDF processing)
- ✅ Google Generative AI (main AI model)
- ✅ PyTorch 2.4.1 (ML framework)
- ✅ Transformers 4.44.2 (HuggingFace models)
- ✅ Sentence Transformers 2.7.0 (embeddings)
- ✅ ChromaDB 0.4.22 (vector store)
- ✅ LangChain ecosystem (document processing)

## Render.com Configuration
```yaml
type: web
name: hackathon-backend
env:
  - PYTHONUNBUFFERED: '1'
  - PYTHONDONTWRITEBYTECODE: '1'
  - PIP_NO_CACHE_DIR: '1'
  - TOKENIZERS_PARALLELISM: 'false'
build:
  buildCommand: |
    pip install --upgrade pip
    pip install --no-cache-dir -r requirements.txt
start:
  startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
```

## Environment Variables to Set in Render
1. `GOOGLE_API_KEY` - Your Google Gemini API key
2. Any other API keys your app needs

## Deployment Steps
1. Push your code to GitHub
2. Connect your GitHub repo to Render.com
3. Set environment variables in Render dashboard
4. Deploy!

## Memory Optimization
- Uses single worker (`--workers 1`)
- Disabled tokenizers parallelism
- No pip cache during build
- ChromaDB version optimized for deployment

## Troubleshooting Tips
- If build fails, check the build logs for specific package issues
- Memory errors: Render free tier has 512MB RAM limit
- For heavy ML models, consider upgrading to paid plan
- Timeout issues: Increase timeout in render.yaml if needed

## API Endpoint
- Health check: `GET /health`
- Main endpoint: `POST /hackrx/run`
- Authentication: Bearer token required
