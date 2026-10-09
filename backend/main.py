
from pathlib import Path
import os
import json

from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai


# ==================================================
# PROJECT PATH
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

ENV_FILE = BASE_DIR / ".env"

load_dotenv(
    dotenv_path=ENV_FILE
)


# ==================================================
# FASTAPI
# ==================================================

app = FastAPI(
    title="Advanced RAG API"
)


# ==================================================
# CORS
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# PROJECT DIRECTORIES
# ==================================================

UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(
    exist_ok=True
)


CHROMA_DIR = BASE_DIR / "chroma_db"


# ==================================================
# EMBEDDING MODEL
# ==================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================================
# CHROMADB
# ==================================================

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_or_create_collection(
    name="documents"
)


# ==================================================
# GEMINI
# ==================================================

gemini_api_key = os.getenv(
    "GEMINI_API_KEY"
)

if not gemini_api_key:

    raise RuntimeError(
        "GEMINI_API_KEY environment variable is missing."
    )


gemini_client = genai.Client(
    api_key=gemini_api_key
)


GEMINI_MODEL = "gemini-3.5-flash-lite"


# ==================================================
# HOME
# ==================================================

@app.get("/")
def home():

    return {
        "message": "Advanced RAG API is running"
    }


# ==================================================
# UPLOAD DOCUMENT
# ==================================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    # --------------------------------------------------
    # Validate filename
    # --------------------------------------------------

    if not file.filename:

        return {
            "message": "No filename provided."
        }


    # --------------------------------------------------
    # Supported file types
    # --------------------------------------------------

    allowed_extensions = {
        ".pdf",
        ".txt"
    }


    extension = Path(
        file.filename
    ).suffix.lower()


    if extension not in allowed_extensions:

        return {
            "message": (
                "Unsupported file type. "
                "Only PDF and TXT files are supported."
            ),
            "filename": file.filename
        }


    # --------------------------------------------------
    # Save uploaded file
    # --------------------------------------------------

    file_path = UPLOAD_DIR / file.filename


    with open(
        file_path,
        "wb"
    ) as buffer:

        buffer.write(
            await file.read()
        )


    # --------------------------------------------------
    # Remove old chunks of same document
    # --------------------------------------------------

    existing = collection.get(
        where={
            "filename": file.filename
        }
    )


    if existing["ids"]:

        collection.delete(
            ids=existing["ids"]
        )


    # --------------------------------------------------
    # Extract document
    # --------------------------------------------------

    documents = extract_document(
        file_path
    )


    if not documents:

        return {
            "message": "Could not extract text from document.",
            "filename": file.filename
        }


    # --------------------------------------------------
    # Create chunks
    # --------------------------------------------------

    all_chunks = []

    all_metadata = []

    chunk_counter = 0


    for document in documents:

        text = document["text"]

        page = document["page"]


        chunks = create_chunks(
            text
        )


        for chunk in chunks:

            all_chunks.append(
                chunk
            )


            metadata = {
                "filename": file.filename,
                "chunk_id": chunk_counter
            }


            # Add page number for PDF
            if page is not None:

                metadata["page"] = page


            all_metadata.append(
                metadata
            )


            chunk_counter += 1


    # --------------------------------------------------
    # Safety check
    # --------------------------------------------------

    if not all_chunks:

        return {
            "message": "No usable text was found in the document.",
            "filename": file.filename
        }


    # --------------------------------------------------
    # Create embeddings
    # --------------------------------------------------

    embeddings = embedding_model.encode(
        all_chunks
    )


    # --------------------------------------------------
    # Create unique IDs
    # --------------------------------------------------

    ids = [

        f"{file.filename}_{i}"

        for i in range(
            len(all_chunks)
        )

    ]


    # --------------------------------------------------
    # Store in ChromaDB
    # --------------------------------------------------

    collection.add(

        ids=ids,

        documents=all_chunks,

        embeddings=embeddings.tolist(),

        metadatas=all_metadata

    )


    # --------------------------------------------------
    # Response
    # --------------------------------------------------

    return {

        "message": "Document stored successfully",

        "filename": file.filename,

        "total_chunks": len(
            all_chunks
        ),

        "embedding_size": len(
            embeddings[0]
        )

    }


# ==================================================
# LIST UPLOADED DOCUMENTS
# ==================================================

@app.get("/documents")
def get_documents():

    results = collection.get(
        include=["metadatas"]
    )


    documents = set()


    for metadata in results["metadatas"]:

        if (
            metadata
            and "filename" in metadata
        ):

            documents.add(
                metadata["filename"]
            )


    return {

        "documents": sorted(
            documents
        )

    }


# ==================================================
# DELETE DOCUMENT
# ==================================================

@app.delete("/documents/{filename}")
def delete_document(
    filename: str
):

    safe_filename = Path(filename).name

    if safe_filename != filename:

        return {
            "message": "Invalid filename."
        }


    # Find all chunks belonging to this document

    existing = collection.get(
        where={
            "filename": safe_filename
        }
    )


    if not existing["ids"]:

        return {
            "message": "Document not found.",
            "filename": safe_filename
        }


    # Delete chunks and embeddings from ChromaDB

    collection.delete(
        ids=existing["ids"]
    )


    # Delete the physical uploaded file

    file_path = UPLOAD_DIR / safe_filename


    if file_path.exists():

        file_path.unlink()


    return {

        "message": "Document deleted successfully.",

        "filename": safe_filename,

        "deleted_chunks": len(
            existing["ids"]
        )

    }


# ==================================================
# VIEW UPLOADED DOCUMENT - NEW FEATURE
# ==================================================

@app.get("/documents/{filename}/view")
def view_document(
    filename: str
):

    # Validate the filename to prevent path traversal.
    safe_filename = Path(filename).name

    if not safe_filename or safe_filename != filename:

        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )


    # Only PDF and TXT files can be viewed.
    extension = Path(
        safe_filename
    ).suffix.lower()

    if extension not in {".pdf", ".txt"}:

        raise HTTPException(
            status_code=415,
            detail="Only PDF and TXT files can be viewed."
        )


    # Serve only files inside the uploads directory.
    file_path = UPLOAD_DIR / safe_filename

    if not file_path.is_file():

        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )


    # Choose the correct content type.
    media_type = (
        "application/pdf"
        if extension == ".pdf"
        else "text/plain; charset=utf-8"
    )


    # Display the document inline in the browser.
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=safe_filename,
        content_disposition_type="inline"
    )


# ==================================================
# SEMANTIC SEARCH
# ==================================================

@app.get("/search")
def search_documents(

    query: str,

    top_k: int = 3

):

    # --------------------------------------------------
    # Create query embedding
    # --------------------------------------------------

    query_embedding = embedding_model.encode(
        [query]
    )


    # --------------------------------------------------
    # Search ChromaDB
    # --------------------------------------------------

    results = collection.query(

        query_embeddings=
        query_embedding.tolist(),

        n_results=top_k

    )


    # --------------------------------------------------
    # Return results
    # --------------------------------------------------

    return {

        "query": query,

        "results":
        results["documents"][0],

        "metadata":
        results["metadatas"][0]

    }


# ==================================================
# RAG QUESTION ANSWERING
# ==================================================

@app.get("/ask")
def ask_question(

    query: str,

    top_k: int = 3,

    conversation_history: str = ""

):

    # --------------------------------------------------
    # Validate question
    # --------------------------------------------------

    if not query.strip():

        return {

            "question": query,

            "answer": "Please enter a question.",

            "sources": []

        }


    # --------------------------------------------------
    # Check whether documents exist
    # --------------------------------------------------

    total_documents = collection.count()


    if total_documents == 0:

        return {

            "question": query,

            "answer": (
                "I could not find any uploaded documents. "
                "Please upload a PDF or TXT document first."
            ),

            "sources": []

        }


    # --------------------------------------------------
    # Create query embedding
    # --------------------------------------------------

    query_embedding = embedding_model.encode(
        [query]
    )


    # --------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------

    results = collection.query(

        query_embeddings=
        query_embedding.tolist(),

        n_results=top_k,

        include=[
            "documents",
            "metadatas",
            "distances"
        ]

    )


    # --------------------------------------------------
    # Retrieved chunks
    # --------------------------------------------------

    retrieved_chunks = results[
        "documents"
    ][0]


    metadata = results[
        "metadatas"
    ][0]


    distances = results[
        "distances"
    ][0]


    # --------------------------------------------------
    # Relevance filtering
    # --------------------------------------------------

    RELEVANCE_THRESHOLD = 1.2


    relevant_chunks = []

    relevant_metadata = []


    for chunk, source_metadata, distance in zip(

        retrieved_chunks,

        metadata,

        distances

    ):

        if distance <= RELEVANCE_THRESHOLD:

            relevant_chunks.append(
                chunk
            )

            relevant_metadata.append(
                source_metadata
            )


    # --------------------------------------------------
    # No sufficiently relevant information
    # --------------------------------------------------

    if not relevant_chunks:

        return {

            "question": query,

            "answer": (
                "I could not find the answer "
                "in the uploaded documents."
            ),

            "sources": []

        }


    # --------------------------------------------------
    # Create context
    # --------------------------------------------------

    context = "\n\n".join(
        relevant_chunks
    )


    # --------------------------------------------------
    # Conversation history
    # --------------------------------------------------

    history_text = ""


    if conversation_history.strip():

        try:

            history = json.loads(
                conversation_history
            )


            if isinstance(history, list):

                recent_history = history[-10:]


                history_lines = []


                for message in recent_history:

                    if not isinstance(message, dict):

                        continue


                    role = message.get(
                        "role",
                        ""
                    )


                    content = message.get(
                        "content",
                        ""
                    )


                    if (
                        role in {
                            "user",
                            "assistant"
                        }
                        and content
                    ):

                        speaker = (
                            "User"
                            if role == "user"
                            else "Assistant"
                        )


                        history_lines.append(

                            f"{speaker}: {content}"

                        )


                history_text = "\\n".join(
                    history_lines
                )


        except json.JSONDecodeError:

            history_text = ""


    # --------------------------------------------------
    # Gemini prompt
    # --------------------------------------------------

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, say:

"I could not find the answer in the uploaded documents."

Do not make up information.

Keep the answer clear and concise.

Conversation history:
{history_text}

Use the conversation history only to understand references
such as "it", "they", "this", or follow-up questions.
The uploaded document context remains the only source of facts.

Context:
{context}

User question:
{query}

Answer:
"""


    # --------------------------------------------------
    # Generate answer
    # --------------------------------------------------

    response = gemini_client.models.generate_content(

        model=GEMINI_MODEL,

        contents=prompt

    )


    # --------------------------------------------------
    # Return answer and relevant sources
    # --------------------------------------------------

    sources = []

    for chunk, source_metadata in zip(
        relevant_chunks,
        relevant_metadata
    ):

        source = {
            "filename": source_metadata.get(
                "filename",
                "Unknown document"
            ),
            "chunk_id": source_metadata.get(
                "chunk_id"
            ),
            "text": chunk
        }

        # Include page number only when available
        if source_metadata.get("page") is not None:

            source["page"] = source_metadata["page"]

        sources.append(source)


    return {

        "question": query,

        "answer": response.text,

        "sources": sources

    }


# ==================================================
# DOCUMENT EXTRACTION
# ==================================================

def extract_document(
    file_path: Path
):

    documents = []


    # ==================================================
    # TXT
    # ==================================================

    if file_path.suffix.lower() == ".txt":

        text = file_path.read_text(
            encoding="utf-8"
        )


        if text.strip():

            documents.append({

                "text": text,

                "page": None

            })


        return documents


    # ==================================================
    # PDF
    # ==================================================

    if file_path.suffix.lower() == ".pdf":

        reader = PdfReader(
            str(file_path)
        )


        for page_number, page in enumerate(

            reader.pages,

            start=1

        ):

            text = page.extract_text()


            if (
                text
                and text.strip()
            ):

                documents.append({

                    "text": text,

                    "page": page_number

                })


        return documents


    # ==================================================
    # UNSUPPORTED FILE
    # ==================================================

    return documents


# ==================================================
# TEXT CHUNKING
# ==================================================

def create_chunks(

    text: str,

    chunk_size: int = 500,

    overlap: int = 50

):

    chunks = []


    start = 0


    while start < len(text):

        end = start + chunk_size


        chunk = text[
            start:end
        ]


        if chunk.strip():

            chunks.append(
                chunk
            )


        start += (
            chunk_size - overlap
        )


    return chunks