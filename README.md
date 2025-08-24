# 📄 Intelligent Document Query & Decision System  

### Official Submission for **Bajaj Finserv Hackathon** 🚀  

---

## 🔹 Overview  
This project implements an **AI-powered Retrieval-Augmented Generation (RAG) pipeline** for intelligent document understanding, contextual retrieval, and decision support. It enables users to query large PDF documents and receive accurate, context-driven answers.  

The system leverages:  
- **RAG architecture** for combining knowledge retrieval with LLM reasoning  
- **LLM API integration** with **Google Gemini** for natural language responses  
- **Efficient document preprocessing** with embeddings stored in **ChromaDB** for fast similarity search  

---

## 🔹 Pipeline Architecture  

1. **Document Ingestion**  
   - PDFs are loaded using `PyPDFDirectoryLoader`.  
   - Each document is split into smaller **chunks**.  
   - Each chunk is assigned a **unique ID** for traceability.  

2. **Query Expansion with HYDE (Hypothetical Document Embeddings)**  
   - A hypothetical answer is generated for the user query.  
   - This synthetic answer is embedded using **GoogleGenerativeEmbeddings**.  

3. **Vector Storage & Retrieval**  
   - Embeddings are stored in **ChromaDB**, enabling efficient similarity search.  
   - On user query, embeddings are compared against stored vectors.  
   - Top **relevant chunks** are retrieved for context.  

4. **Context Summarization (if required)**  
   - Retrieved chunks are summarized for concise context.  

5. **LLM Answer Generation (Gemini)**  
   - Final query + context are passed to **Google Gemini**.  
   - Gemini generates a **context-aware, natural language response**.  
   - **Only Gemini** is used for the final output (ensuring consistency).  

---

## 🔹 Tech Stack  
- **Python** – Core implementation  
- **LangChain** – Document loading, chunking, and RAG orchestration  
- **ChromaDB** – Vector database for embeddings  
- **GoogleGenerativeAI (Gemini)** – Answer generation  
- **GoogleGenerativeEmbeddings** – Query & document embeddings  
- **HYDE** – Hypothetical Document Embeddings for enhanced retrieval  

---

## 🔹 Key Features  
✅ **RAG-based Pipeline** – Ensures precise, context-driven responses  
✅ **HYDE Query Expansion** – Improves retrieval accuracy  
✅ **LLM API Integration** – Seamless Gemini integration for answer generation  
✅ **Efficient PDF Handling** – Automatic chunking and ID assignment  
✅ **Scalable Vector Search** – ChromaDB ensures fast, relevant retrieval  

---

## 🔹 Workflow Diagram  

```mermaid
flowchart TD
    A[PDF Documents] --> B[PyPDFDirectoryLoader]
    B --> C[Chunking + Unique IDs]
    C --> D[ChromaDB Storage]

    E[User Query] --> F[HYDE - Hypothetical Answer]
    F --> G[GoogleGenerativeEmbeddings]
    G --> H[ChromaDB Retrieval]
    H --> I[Top Relevant Chunks]
    I --> J[Context Summarization]

    J --> K[Final Prompt + Context]
    K --> L[Gemini LLM]
    L --> M[Final Response]
