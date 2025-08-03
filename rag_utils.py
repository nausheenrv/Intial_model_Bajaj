from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import os

CHROMA_PATH = "./chroma_db"

# Create Chroma and embedding function
embedding_function = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_function,
)

# Function to process questions using RAG
# This is a simplified version
async def process_questions(text, questions):
    results = []

    # Add the document text to the Chroma database (for illustrative purposes)
    # In a real scenario, you might split text into chunks and store separately
    document_chunks = [{'id': "1", 'text': text}]  # Simplified
    for chunk in document_chunks:
        db.add_documents([chunk], ids=[chunk['id']])

    # Process each question
    for question_text in questions:
        # Use the text splitter and embedding
        hyde_llm = OllamaLLM(model="Mistral", timeout=30)
        hypothetical_answer = hyde_llm.invoke(f"Answer this question briefly: {question_text}")

        # Retrieve documents
        results = db.similarity_search_with_score(hypothetical_answer, k=5)

        # Prepare context and prompt
        PROMPT_TEMPLATE = """
        Answer the question based only on the following context:
        {context}
        ---
        Answer the question: {question}
        """
        top_docs = [doc['text'] for doc, _ in results]
        context_text = "\n\n---\n\n".join(top_docs)
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=question_text)

        # Generate final answer
        model = OllamaLLM(model="Mistral", timeout=60)
        response = model.invoke(prompt)
        results.append(response)

    return results

