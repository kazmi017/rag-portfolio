import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import pymupdf

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
db_client = chromadb.Client()
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def chunk_text(text, chunk_size):
    text = text.strip()
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def build_index(pdf_path):
    doc = pymupdf.open(pdf_path)
    all_chunks = []
    chunk_pages = []

    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        page_text = page_text.replace("O N L I N E V E R S I O N", " ")
        page_text = page_text.replace("Building Regulations 2010 \nApproved Document B Volume 1, 2019 edition", " ")
        page_text = page_text.replace("\n", " ")
        page_chunks = chunk_text(page_text, chunk_size=1000)
        for chunk in page_chunks:
            all_chunks.append(chunk)
            chunk_pages.append(page_num + 1)

    embeddings = embedder.encode(all_chunks).tolist()

    existing = [c.name for c in db_client.list_collections()]
    if "active_docs" in existing:
        db_client.delete_collection("active_docs")

    collection = db_client.create_collection("active_docs")
    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(all_chunks))],
        metadatas=[{"page": p} for p in chunk_pages]
    )

    print(f"Indexed {len(all_chunks)} chunks from {doc.page_count} pages: {pdf_path}")
    return collection

def reset_qa_history():
    existing = [c.name for c in db_client.list_collections()]
    if "qa_history" in existing:
        db_client.delete_collection("qa_history")
    return db_client.create_collection("qa_history")

def answer_question(collection, qa_history, question, qa_counter):
    question_embedding = embedder.encode([question]).tolist()

    related_history = ""
    if qa_history.count() > 0:
        history_results = qa_history.query(query_embeddings=question_embedding, n_results=1)
        related_history = history_results["documents"][0][0]

    results = collection.query(query_embeddings=question_embedding, n_results=4)
    retrieved_chunks = results["documents"][0]
    retrieved_pages = [m["page"] for m in results["metadatas"][0]]
    distances = results["distances"][0]
    
    DISTANCE_THRESHOLD = 1.5

    if min(distances) > DISTANCE_THRESHOLD:
        answer = "I couldn't find anything in this document that answers that question."
        qa_history.add(
            documents=[f"Q: {question}\nA: {answer}"],
            embeddings=embedder.encode([f"Q: {question}\nA: {answer}"]).tolist(),
            ids=[f"qa_{qa_counter}"]
        )
        return answer, [], distances

    context = "\n\n---\n\n".join(
        f"[Page {page}]\n{chunk}" for chunk, page in zip(retrieved_chunks, retrieved_pages)
    )

    response = deepseek_client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": "Answer only using the provided context. If the context doesn't contain the answer, say so clearly instead of guessing. Cite the page number(s) you used. Be concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nPrevious related Q&A (may or may not be relevant):\n{related_history}\n\nQuestion: {question}"}
        ]
    )
    answer = response.choices[0].message.content

    qa_history.add(
        documents=[f"Q: {question}\nA: {answer}"],
        embeddings=embedder.encode([f"Q: {question}\nA: {answer}"]).tolist(),
        ids=[f"qa_{qa_counter}"]
    )

    return answer, retrieved_pages, distances