# Document Question-Answering API

This project lets you ask questions about a PDF document. It reads the document, divides it into meaningful text sections, turns those sections into searchable numerical representations, and uses the closest sections as context for an AI-generated answer.

The application starts with `fire_safety.pdf` by default. A different PDF can be uploaded while the application is running, and future questions will use the newly uploaded document.

## How It Works

1. The application opens a PDF and extracts its text page by page.
2. The text is split into overlapping sections so that useful context is preserved between sections.
3. Each section is converted into an embedding using `all-MiniLM-L6-v2`.
4. ChromaDB stores the sections and their page numbers for similarity search.
5. A question is also converted into an embedding, and the most relevant sections are retrieved.
6. DeepSeek receives the retrieved context and creates a concise answer.
7. The answer includes relevant source pages when the document contains a suitable match.

If the document does not contain relevant information, the application reports that instead of asking the language model to guess.

## Project Files

| File | Purpose |
| --- | --- |
| `main.py` | Creates the FastAPI application and exposes the upload and question endpoints. |
| `rag_engine.py` | Handles PDF extraction, text chunking, embeddings, document search, and answer generation. |
| `fire_safety.pdf` | The document indexed when the server starts. |
| `.env` | Stores the DeepSeek API key locally. This file is ignored by Git. |

## Requirements

- Python 3.10 or newer
- A DeepSeek API key
- The packages listed in the installation command below

## Setup

Open a terminal in the project folder and create a virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install fastapi uvicorn python-multipart python-dotenv sentence-transformers chromadb openai PyMuPDF
```

Create a file named `.env` in the project folder:

```env
DEEPSEEK_API_KEY=your_api_key_here
```

The API key is read when the application starts. Do not commit `.env` or share the key.

## Run the Application

Start the API with Uvicorn:

```powershell
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

The first startup may take longer because the embedding model is downloaded and the default PDF is indexed.

## API Usage

### Ask a question

Send a question to `/ask`:

```text
http://127.0.0.1:8000/ask?question=What%20does%20the%20document%20say%20about%20fire%20doors?
```

Example response:

```json
{
  "question": "What does the document say about fire doors?",
  "answer": "...",
  "source_pages": [12, 13],
  "match_distances": [0.42, 0.61, 0.88, 1.02]
}
```

`source_pages` identifies the pages used as context. `match_distances` contains the similarity-search distances; lower values indicate closer matches.

### Upload a PDF

Upload a new PDF to `/upload`:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/upload" -F "file=@my-document.pdf"
```

After a successful upload, the new document becomes active and the previous question history is cleared. The uploaded file is saved in the project folder with an `uploaded_` prefix.

## Notes

- The application keeps its ChromaDB data in memory while the process is running.
- Restarting the server rebuilds the index from `fire_safety.pdf`.
- Only the active document is searched at a time.
- The `/ask` endpoint currently uses a query-string parameter rather than a JSON request body.
- For production use, add authentication, file-type and file-size validation, persistent storage, and stronger error handling before exposing the service publicly.

## Stopping the Server

Press `Ctrl+C` in the terminal running Uvicorn.