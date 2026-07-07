# Resume-RAG-Project

## Overview
Resume-RAG-Project is a retrieval-augmented generation application for answering questions from resume PDFs using local embeddings and the Google Gemini API.

## Architecture
The pipeline flows as follows:

PDF -> Chunking -> Embedding -> Vector Store -> Retriever -> Prompt Builder -> Gemini -> Answer

## Features
- Load one or more resume PDFs from the data directory
- Chunk and embed resume content
- Store embeddings in ChromaDB
- Retrieve context for a user question
- Generate answers using Gemini without hallucinating beyond the retrieved context
- Handle missing files, empty folders, invalid API keys, and network issues gracefully

## Project Structure
```text
Resume-RAG-Project/
├── data/
├── ground_truth/
├── src/
│   ├── data_processing/
│   ├── evaluation/
│   ├── rag/
│   └── ui/
├── .env.example
├── .gitignore
├── main.py
├── requirements.txt
└── README.md
```

## Installation
1. Create and activate a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy [.env.example](.env.example) to .env and fill in the values.
4. Place one or more PDF resumes in the data folder.
5. Run the CLI:
   ```bash
   python main.py
   ```

## Environment Variables
```env
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL_NAME=gemini-2.5-flash
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
CHROMA_DB_PATH=./.chromadb
HF_HOME=./.cache/huggingface
TOP_K=5
```

## Sample Queries
- What is the candidate's full name?
- What are the candidate's technical skills?
- Which projects mention Python?

## Dependencies
- ChromaDB for vector storage
- PyPDF for PDF extraction
- SentenceTransformers for embeddings
- python-dotenv for environment loading
- google-genai for Gemini access
- Streamlit for the optional UI

## Notes
- Do not commit the .env file.
- The application will return a friendly message when no relevant context is found or when Gemini is unavailable.