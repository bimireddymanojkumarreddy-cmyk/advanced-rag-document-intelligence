# Advanced RAG Document Intelligence

An AI-powered document question-answering application that lets users upload documents and ask questions about their contents using Retrieval-Augmented Generation (RAG).

## 🚀 Live Demo

**Frontend:** [Open the application](https://bimireddymanojkumarreddy-cmyk.github.io/advanced-rag-document-intelligence/)

## ✨ Features

* 📄 Upload documents and query their contents.
* 💬 Ask questions using a chat-style interface.
* 🔎 Retrieve relevant document content to help answer questions.
* 📚 Display source information when available.
* 🧠 Generate answers using a large language model.
* 🌐 Deploy the frontend on GitHub Pages and the backend on Render.
* 🎨 Responsive interface with dark and light themes.

## 🛠️ Technology Stack

**Frontend**

* HTML
* CSS
* JavaScript

**Backend**

* Python
* FastAPI
* ChromaDB for vector storage
* ONNX Runtime and MiniLM-based embeddings
* Google Gemini API for answer generation
* PyPDF for PDF text extraction

## 🏗️ Architecture

1. **Document upload:** The user uploads a supported document.
2. **Text extraction:** The backend extracts text from the document.
3. **Chunking and embeddings:** The text is divided into chunks and converted into vector embeddings.
4. **Vector storage:** Chunks and their metadata are stored in ChromaDB.
5. **Retrieval:** A question is used to retrieve relevant document chunks.
6. **Answer generation:** The retrieved context is passed to the language model to generate an answer.
7. **Response:** The frontend displays the answer and available source information.

## 📁 Project Structure

```text
advanced-rag-document-intelligence/
├── backend/
│   └── main.py
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── index.html
├── script.js
├── style.css
├── requirements.txt
└── README.md
```

The root-level frontend files are used by GitHub Pages. The `frontend/` directory contains the original frontend files.

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/bimireddymanojkumarreddy-cmyk/advanced-rag-document-intelligence.git
cd advanced-rag-document-intelligence
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file and configure the Gemini API key using the environment variable name expected by `backend/main.py`.

**Security:** Never commit API keys or other secrets to GitHub.

### 5. Start the backend

Run the FastAPI application using the module and settings configured in `backend/main.py`. For example, if the application object is named `app`:

```bash
uvicorn backend.main:app --reload
```

The backend API documentation is typically available at:

```text
http://127.0.0.1:8000/docs
```

## ☁️ Deployment

* **Frontend:** GitHub Pages
* **Backend:** Render
* **Language model:** Google Gemini API

The deployed frontend communicates with the backend API. When running locally, configure the frontend API URL to point to your local backend.

## 🎯 Project Objective

This project demonstrates the practical use of Retrieval-Augmented Generation, vector databases, semantic search, document processing, embeddings, and large language models to build an interactive document intelligence application.

## 🔮 Future Improvements

* Improve retrieval quality and chunking strategies.
* Add document-level filtering and better source navigation.
* Add automated evaluation of answer relevance and retrieval quality.
* Improve error handling and user feedback.
* Add authentication and per-user document isolation.

## 👨‍💻 Author

**Manoj Kumar Reddy**

Built as a learning and portfolio project focused on AI, machine learning, and generative AI applications.
