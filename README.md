# Resume-RAG-Project

## Overview
Resume-RAG-Project is a Retrieval-Augmented Generation (RAG) application that generates interview questions from resume content using Google Gemini and LangChain.

## Features
- Upload resume in PDF format
- Extract resume content
- Generate interview questions
- Compare generated questions with ground truth
- Evaluate similarity score

## Project Structure
```
Resume-RAG-Project/
│── data/
│── ground_truth/
│── src/
│── .env.example
│── .gitignore
│── main.py
│── requirements.txt
│── README.md
```

## Requirements
- Python 3.10+
- Google Gemini API Key

## Installation

1. Clone the repository
```bash
git clone <repository-url>
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file
```env
GOOGLE_API_KEY=your_api_key
MODEL_NAME=gemini-2.5-flash
CHROMA_DB_PATH=./data/chroma_db
```

4. Run the application
```bash
python main.py
```

## Technologies Used
- Python
- LangChain
- Google Gemini
- ChromaDB
- Streamlit
- RAG

## Note
Do not upload the `.env` file to GitHub. Use `.env.example` as a reference.