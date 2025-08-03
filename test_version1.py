from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from transformers import pipeline
import sys
import mlflow
import mlflow.sklearn
from datetime import datetime

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
    
    # Initialize MLflow run
    mlflow.set_experiment("RAG_Pipeline_Tracking")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    with mlflow.start_run(run_name=f"RAG_Query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        params = {
            "query": query_text,
            "chunk_size": 800,
            "chunk_overlap": 80,
            "embedding_model": "nomic-embed-text",
            "hyde_model": "gemma3",
            "main_llm_model": "Mistral",
            "retrieval_k": 5,
            "context_length_threshold": 3000
        }
        mlflow.log_params(params)
        
        # Start total timing
        total_start = time.time()
        
        # Initialize embedding function
        embedding_function = get_embedding_function()
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_function,
        )
        
        # Query classification (optional)
        classification_start = time.time()
        query_type = classify_query(query_text)
        classification_end = time.time()
        classification_time = classification_end - classification_start
        
        print(f"Query type: {query_type}")
        
        # HyDE Generation
        hyde_start = time.time()
        hyde_llm = OllamaLLM(model="gemma3", timeout=30) 
        hypothetical_answer = hyde_llm.invoke(f"Answer this question briefly (hypothetical answer, even if not grounded): {query_text}")
        hyde_end = time.time()
        hyde_time = hyde_end - hyde_start
        
        print(f"HyDE hypothetical answer:\n{hypothetical_answer}\n")
        
        # Vector Retrieval
        retrieval_start = time.time()
        results = db.similarity_search_with_score(hypothetical_answer, k=5)
        retrieval_end = time.time()
        retrieval_time = retrieval_end - retrieval_start
        
        # Context Processing and Prompt Creation
        prompt_start = time.time()
        
        PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---
Answer the question: {question}
"""

        top_docs = [doc for doc, _ in results]
        context_text = "\n\n---\n\n".join([doc.page_content for doc in top_docs])
        original_context_length = len(context_text)
        
        # Context summarization if needed
        summarization_time = 0
        context_was_summarized = False
        if len(context_text) > 3000:
            summary_start = time.time()
            summary = summarize_context(context_text)
            summary_end = time.time()
            summarization_time = summary_end - summary_start
            context_was_summarized = True
            final_context = summary
        else:
            final_context = context_text

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=final_context, question=query_text)
        prompt_end = time.time()
        prompt_time = prompt_end - prompt_start
        
        # Final LLM Generation
        llm_start = time.time()
        model = OllamaLLM(model="Mistral", timeout=60)
        response_text = model.invoke(prompt)
        llm_end = time.time()
        llm_generation_time = llm_end - llm_start
        
        # Total time calculation
        total_end = time.time()
        total_time = total_end - total_start
        
        # Extract sources and similarity scores
        sources = [doc.metadata.get("id", None) for doc, _ in results]
        similarity_scores = [float(score) for _, score in results]
        avg_similarity_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        
        # Log timing metrics
        timing_metrics = {
            "classification_time_sec": classification_time,
            "hyde_generation_time_sec": hyde_time,
            "vector_retrieval_time_sec": retrieval_time,
            "prompt_creation_time_sec": prompt_time,
            "llm_generation_time_sec": llm_generation_time,
            "summarization_time_sec": summarization_time,
            "total_pipeline_time_sec": total_time
        }
        mlflow.log_metrics(timing_metrics)
        
        # Log quality and context metrics
        quality_metrics = {
            "num_retrieved_docs": len(top_docs),
            "original_context_length": original_context_length,
            "final_context_length": len(final_context),
            "context_was_summarized": int(context_was_summarized),
            "avg_similarity_score": avg_similarity_score,
            "min_similarity_score": min(similarity_scores) if similarity_scores else 0,
            "max_similarity_score": max(similarity_scores) if similarity_scores else 0,
            "response_length": len(response_text)
        }
        mlflow.log_metrics(quality_metrics)
        
        # Log additional information as artifacts
        mlflow.log_text(hypothetical_answer, "hypothetical_answer.txt")
        mlflow.log_text(response_text, "final_response.txt")
        mlflow.log_text("\n".join(sources), "retrieved_sources.txt")
        
        # Log individual similarity scores
        for i, score in enumerate(similarity_scores):
            mlflow.log_metric(f"similarity_score_doc_{i+1}", score)
        
        # Log timing breakdown as tags
        mlflow.set_tags({
            "query_type": query_type,
            "hyde_enabled": "true",
            "summarization_used": str(context_was_summarized),
            "pipeline_version": "v1.0"
        })
        
        # Print timing summary
        print(f"\n=== Timings ===")
        print(f"Query Classification:     {classification_time:.2f} sec")
        print(f"HyDE Generation Time:     {hyde_time:.2f} sec")
        print(f"Vector Retrieval Time:    {retrieval_time:.2f} sec")
        print(f"Prompt Creation Time:     {prompt_time:.2f} sec")
        if context_was_summarized:
            print(f"Context Summarization:    {summarization_time:.2f} sec")
        print(f"LLM Answer Generation:    {llm_generation_time:.2f} sec")
        print(f"Total Pipeline Latency:   {total_time:.2f} sec")
        
        print(f"\n=== Quality Metrics ===")
        print(f"Average Similarity Score: {avg_similarity_score:.4f}")
        print(f"Context Length: {original_context_length} -> {len(final_context)}")
        print(f"Response Length: {len(response_text)} characters")

        print("Response:", response_text)
        print("Sources:", sources)

        return response_text, sources

def run_experiment_batch(queries_list):
    """Run multiple queries and compare performance"""
    
    mlflow.set_experiment("RAG_Batch_Experiments")
    
    # Load documents once
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    
    experiment_start = time.time()
    
    with mlflow.start_run(run_name=f"Batch_Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log batch parameters
        batch_params = {
            "num_queries": len(queries_list),
            "total_chunks": len(chunks),
            "experiment_type": "batch_comparison"
        }
        mlflow.log_params(batch_params)
        
        total_times = []
        hyde_times = []
        llm_times = []
        
        for i, query in enumerate(queries_list):
            print(f"\n--- Processing Query {i+1}/{len(queries_list)} ---")
            print(f"Query: {query}")
            
            # This will create individual runs for each query
            response, sources = query_rag(query, chunks)
            
        experiment_end = time.time()
        
        # Log batch summary metrics
        mlflow.log_metric("total_experiment_time_sec", experiment_end - experiment_start)
        mlflow.log_metric("avg_time_per_query", (experiment_end - experiment_start) / len(queries_list))

if __name__ == "__main__":
    
    # Example batch experiment
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        sample_queries = [
            "What is machine learning?",
            "How does neural network training work?",
            "Explain the difference between supervised and unsupervised learning",
            "What are the applications of deep learning?"
        ]
        run_experiment_batch(sample_queries)
    
    # Single query processing
    elif len(sys.argv) > 1:
        documents = load_documents()
        chunks = split_documents(documents)
        print("Sample chunk content:")
        print(chunks[0].page_content)
        add_to_chroma(chunks)
        
        query_text = sys.argv[1]
        print(f"\nProcessing query: {query_text}")
        query_rag(query_text, chunks)
    
    else:
        print("Usage:")
        print("  Single query: python extract.py 'Your question here'")
        print("  Batch experiment: python extract.py --batch")