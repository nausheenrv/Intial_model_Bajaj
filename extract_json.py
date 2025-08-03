

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from transformers import pipeline
import sys
import json

CHROMA_PATH = "chroma_db"

def load_documents():
    loader = PyPDFDirectoryLoader("pdfs")
    return loader.load()

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def get_embedding_function():
    return OllamaEmbeddings(model="nomic-embed-text")

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_query(query):
    labels = ["factual", "definition", "causal", "how-to", "comparison", "summarization"]
    result = classifier(query, candidate_labels=labels)
    return result["labels"][0]

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing items in db: {len(existing_ids)}")

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

    if new_chunks:
        db.add_documents(new_chunks, ids=new_chunk_ids)
        print(f"Added {len(new_chunks)} new chunks.")
    else:
        print("No new chunks to add.")

def summarize_context(context_text):
    summarizer = OllamaLLM(model="Mistral", timeout=30)
    return summarizer.invoke(f"Summarize the following content concisely:\n{context_text}")

def query_rag(query_text: str, chunks):
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
    )

    query_type = classify_query(query_text)
    print(f"Query type: {query_type}")

    hyde_llm = OllamaLLM(model="Mistral", timeout=30)
    hypothetical_answer = hyde_llm.invoke(
        f"Answer this question briefly (hypothetical answer, even if not grounded): {query_text}"
    )
    print(f"HyDE hypothetical answer:\n{hypothetical_answer}\n")

    results = db.similarity_search_with_score(hypothetical_answer, k=5)

    PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---
Answer the question: {question}
"""

    top_docs = [doc for doc, _ in results]
    context_text = "\n\n---\n\n".join([doc.page_content for doc in top_docs])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    summary = summarize_context(context_text)
    prompt = prompt_template.format(context=summary, question=query_text)

    model = OllamaLLM(model="Mistral", timeout=60)
    response_text = model.invoke(prompt)

    response_data = {
        "query": query_text,
        "response": response_text,
        "sources": [doc.metadata.get("id", "N/A") for doc in top_docs]
    }

    return response_data

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