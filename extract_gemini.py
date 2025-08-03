from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
import google.generativeai as genai
import os
import sys
import json
import time
CHROMA_PATH = "chroma_db"
os.environ["GOOGLE_API_KEY"] = "AIzaSyCzK5gdfDGmPcQENRHdC6AhDfMh3gkwAWY"  

# === Gemini API wrapper ===
def invoke_gemini(prompt, model="gemini-1.5-pro"):
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    llm = genai.GenerativeModel(model)
    response = llm.generate_content(prompt)
    return response.text.strip()

# === Load PDFs ===
def load_documents():
    loader = PyPDFDirectoryLoader("pdfs")
    return loader.load()

# === Split into Chunks ===
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

# === Use Ollama for embeddings ===
def get_embedding_function():
    return OllamaEmbeddings(model="nomic-embed-text")

# === Use Gemini for classification ===
def classify_query(query):
    prompt = (
        f"Classify this query into one of the following categories:\n"
        f"['factual', 'definition', 'causal', 'how-to', 'comparison', 'summarization']\n"
        f"Query: \"{query}\"\n"
        f"Respond with just the category label."
    )
    return invoke_gemini(prompt)

# === Add to Chroma Vector DB ===
def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
 

    last_page_id = None
    current_chunk_index = 0
    new_chunks = []
    new_chunk_ids = []

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
        last_page_id = current_page_id

        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_chunk_ids.append(chunk_id)

  

# === Use Gemini to summarize context ===
def summarize_context(context_text):
    return invoke_gemini(f"Summarize the following context concisely:\n{context_text}")

# === Core RAG Pipeline ===
def query_rag(query_text: str, chunks):
    total_start = time.time()
    db = Chroma(
        persist_directory=CHROMA_PATH,

        embedding_function=get_embedding_function(),
    )

    query_type = classify_query(query_text)

    # Use Gemini to generate HyDE-style hypothetical answer
    hypothetical_answer = invoke_gemini(
        f"Generate a brief, hypothetical answer to the following question (even if not grounded):\n{query_text}"
    )
 

    # Semantic Search in Chroma
    results = db.similarity_search_with_score(hypothetical_answer, k=5)
 
    # Format retrieved docs
    top_docs = [doc for doc, _ in results]
    context_text = "\n\n---\n\n".join([doc.page_content for doc in top_docs])


    # Final prompt
    PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---
Answer the question: {question}
"""
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    prompt_end = time.time()
    if len(context_text) > 3000:
       summary = summarize_context(context_text)
    else:
      summary = context_text
    prompt = prompt_template.format(context=summary, question=query_text)  


    # Final answer using Gemini
    response_text = invoke_gemini(prompt)



    # Pack result
    response_data = {
        "questions": query_text,
        "answers": response_text,
   
    }

    return response_data

# === Entry Point ===
if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)
    print("Sample chunk content:")

    print(chunks[0].page_content)
    add_to_chroma(chunks)

    if len(sys.argv) > 1:
        query_text = sys.argv[1]
        print(f"\nProcessing query: {query_text}")
        response = query_rag(query_text, chunks)
        print("\nFinal Answer:")
        print(json.dumps(response, indent=2))
    else:
        print("Please pass a query as a command-line argument.")
