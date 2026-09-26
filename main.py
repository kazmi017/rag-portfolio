from fastapi import FastAPI, UploadFile, File
import rag_engine

app = FastAPI()

collection = rag_engine.build_index("fire_safety.pdf")
qa_history = rag_engine.reset_qa_history()
qa_counter = 0

print("Setup complete. Server ready with default document (fire_safety.pdf).")

@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    global collection, qa_history, qa_counter

    save_path = f"uploaded_{file.filename}"
    with open(save_path, "wb") as f:
        f.write(file.file.read())

    collection = rag_engine.build_index(save_path)
    qa_history = rag_engine.reset_qa_history()
    qa_counter = 0

    return {"message": f"Now using uploaded document: {file.filename}"}

@app.get("/ask")
def ask(question: str):
    global qa_counter

    answer, pages, distances = rag_engine.answer_question(collection, qa_history, question, qa_counter)
    qa_counter += 1

    return {
        "question": question,
        "answer": answer,
        "source_pages": pages,
        "match_distances": distances
    }