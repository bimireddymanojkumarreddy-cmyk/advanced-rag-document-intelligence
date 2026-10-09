
from pathlib import Path
import os
import json

import numpy as np
import chromadb
import onnxruntime as ort

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download
from pypdf import PdfReader
from tokenizers import Tokenizer
from google import genai


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ENV_FILE = BASE_DIR / ".env"
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(dotenv_path=ENV_FILE)


# ==================================================
# FASTAPI
# ==================================================

app = FastAPI(title="Advanced RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# ONNX EMBEDDING MODEL
# ==================================================

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="onnx/model.onnx",
)

tokenizer_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="tokenizer.json",
)

tokenizer = Tokenizer.from_file(tokenizer_path)
tokenizer.enable_truncation(max_length=256)
tokenizer.enable_padding()

embedding_session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"],
)

print("ONNX model inputs:")
for item in embedding_session.get_inputs():
    print(item.name, item.shape, item.type)

print("ONNX model outputs:")
for item in embedding_session.get_outputs():
    print(item.name, item.shape, item.type)


# ==================================================
# CREATE EMBEDDINGS
# ==================================================

def create_embeddings(texts):
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    encoded = tokenizer.encode_batch(
        [str(text) for text in texts]
    )

    input_ids = np.array(
        [item.ids for item in encoded],
        dtype=np.int64,
    )

    attention_mask = np.array(
        [item.attention_mask for item in encoded],
        dtype=np.int64,
    )

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    input_names = {
        item.name for item in embedding_session.get_inputs()
    }

    if "token_type_ids" in input_names:
        model_inputs["token_type_ids"] = np.array(
            [item.type_ids for item in encoded],
            dtype=np.int64,
        )

    outputs = embedding_session.run(
        None,
        model_inputs,
    )

    # The standard MiniLM ONNX encoder should return
    # token embeddings with shape (batch, tokens, 384).
    token_embeddings = outputs[0]

    if token_embeddings.ndim != 3:
        raise RuntimeError(
            "Unexpected ONNX output shape: "
            f"{token_embeddings.shape}. "
            "Expected token-level embeddings."
        )

    if token_embeddings.shape[-1] != 384:
        raise RuntimeError(
            "Unexpected embedding dimension: "
            f"{token_embeddings.shape[-1]}. "
            "Expected 384."
        )

    mask = attention_mask[:, :, np.newaxis].astype(
        np.float32
    )

    token_embeddings = token_embeddings.astype(
        np.float32
    )

    summed = np.sum(
        token_embeddings * mask,
        axis=1,
    )

    counts = np.maximum(
        mask.sum(axis=1),
        1e-9,
    )

    sentence_embeddings = summed / counts

    norms = np.linalg.norm(
        sentence_embeddings,
        axis=1,
        keepdims=True,
    )

    sentence_embeddings = sentence_embeddings / np.maximum(
        norms,
        1e-12,
    )

    if not np.isfinite(sentence_embeddings).all():
        raise RuntimeError(
            "Embedding generation produced invalid values."
        )

    return sentence_embeddings.astype(np.float32)


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

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is missing."
    )

gemini_client = genai.Client(
    api_key=gemini_api_key
)

GEMINI_MODEL = "gemini-3.5-flash-lite"


# ==================================================
# HOME / HEALTH CHECK
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
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    safe_filename = Path(file.filename).name

    if safe_filename != file.filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    extension = Path(safe_filename).suffix.lower()

    if extension not in {".pdf", ".txt"}:
        raise HTTPException(
            status_code=415,
            detail="Only PDF and TXT files are supported.",
        )

    file_path = UPLOAD_DIR / safe_filename

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        file_path.write_bytes(content)

        documents = extract_document(file_path)

        if not documents:
            file_path.unlink(missing_ok=True)

            raise HTTPException(
                status_code=422,
                detail="Could not extract text from document.",
            )

        all_chunks = []
        all_metadata = []

        for document in documents:
            chunks = create_chunks(document["text"])

            for chunk in chunks:
                metadata = {
                    "filename": safe_filename,
                    "chunk_id": len(all_chunks),
                }

                if document["page"] is not None:
                    metadata["page"] = document["page"]

                all_chunks.append(chunk)
                all_metadata.append(metadata)

        if not all_chunks:
            file_path.unlink(missing_ok=True)

            raise HTTPException(
                status_code=422,
                detail="No usable text was found in the document.",
            )

        # Generate one embedding per chunk.
        embeddings = create_embeddings(all_chunks).tolist()

        ids = [
            f"{safe_filename}_{i}"
            for i in range(len(all_chunks))
        ]

        # Replace existing indexed chunks for this filename.
        existing = collection.get(
            where={"filename": safe_filename}
        )

        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        collection.add(
            ids=ids,
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadata,
        )

        return {
            "message": "Document stored successfully",
            "filename": safe_filename,
            "total_chunks": len(all_chunks),
            "embedding_size": len(embeddings[0]),
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(f"Document upload error: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Document processing failed. Check the server logs.",
        ) from exc

    finally:
        await file.close()


# ==================================================
# LIST UPLOADED DOCUMENTS
# ==================================================

@app.get("/documents")
def get_documents():
    results = collection.get(
        include=["metadatas"]
    )

    documents = set()

    for metadata in results["metadatas"] or []:
        if metadata and metadata.get("filename"):
            documents.add(metadata["filename"])

    return {
        "documents": sorted(documents)
    }


# ==================================================
# DELETE DOCUMENT
# ==================================================

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    safe_filename = Path(filename).name

    if not safe_filename or safe_filename != filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    existing = collection.get(
        where={"filename": safe_filename}
    )

    if not existing["ids"]:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    collection.delete(ids=existing["ids"])

    file_path = UPLOAD_DIR / safe_filename

    if file_path.is_file():
        file_path.unlink()

    return {
        "message": "Document deleted successfully.",
        "filename": safe_filename,
        "deleted_chunks": len(existing["ids"]),
    }


# ==================================================
# VIEW DOCUMENT
# ==================================================

@app.get("/documents/{filename}/view")
def view_document(filename: str):
    safe_filename = Path(filename).name

    if not safe_filename or safe_filename != filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    extension = Path(safe_filename).suffix.lower()

    if extension not in {".pdf", ".txt"}:
        raise HTTPException(
            status_code=415,
            detail="Only PDF and TXT files can be viewed.",
        )

    file_path = UPLOAD_DIR / safe_filename

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    media_type = (
        "application/pdf"
        if extension == ".pdf"
        else "text/plain; charset=utf-8"
    )

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=safe_filename,
        content_disposition_type="inline",
    )


# ==================================================
# SEMANTIC SEARCH
# ==================================================

@app.get("/search")
def search_documents(
    query: str,
    top_k: int = 3,
):
    query = query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    top_k = max(1, min(top_k, 10))

    if collection.count() == 0:
        return {
            "query": query,
            "results": [],
            "metadata": [],
        }

    query_embedding = create_embeddings(
        [query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
    )

    return {
        "query": query,
        "results": results["documents"][0] or [],
        "metadata": results["metadatas"][0] or [],
    }


# ==================================================
# RAG QUESTION ANSWERING
# ==================================================

@app.get("/ask")
def ask_question(
    query: str,
    top_k: int = 3,
    conversation_history: str = "",
):
    query = query.strip()

    if not query:
        return {
            "question": query,
            "answer": "Please enter a question.",
            "sources": [],
        }

    total_chunks = collection.count()

    if total_chunks == 0:
        return {
            "question": query,
            "answer": (
                "I could not find any uploaded documents. "
                "Please upload a PDF or TXT document first."
            ),
            "sources": [],
        }

    top_k = max(1, min(top_k, 10))

    query_embedding = create_embeddings(
        [query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, total_chunks),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved_chunks = results["documents"][0] or []
    metadata = results["metadatas"][0] or []
    distances = results["distances"][0] or []

    relevance_threshold = 1.2

    relevant_chunks = []
    relevant_metadata = []

    for chunk, source_metadata, distance in zip(
        retrieved_chunks,
        metadata,
        distances,
    ):
        if distance <= relevance_threshold:
            relevant_chunks.append(chunk)
            relevant_metadata.append(source_metadata)

    if not relevant_chunks:
        return {
            "question": query,
            "answer": (
                "I could not find the answer "
                "in the uploaded documents."
            ),
            "sources": [],
        }

    context = "\n\n".join(relevant_chunks)

    # Safely format conversation history.
    history_text = ""

    if conversation_history.strip():
        try:
            history = json.loads(conversation_history)

            if isinstance(history, list):
                history_lines = []

                for message in history[-10:]:
                    if not isinstance(message, dict):
                        continue

                    role = message.get("role")
                    content = message.get("content")

                    if (
                        role in {"user", "assistant"}
                        and isinstance(content, str)
                        and content.strip()
                    ):
                        speaker = (
                            "User"
                            if role == "user"
                            else "Assistant"
                        )

                        history_lines.append(
                            f"{speaker}: {content}"
                        )

                history_text = "\n".join(history_lines)

        except (json.JSONDecodeError, TypeError):
            history_text = ""

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information
provided in the document context below.

If the answer cannot be found in the context, say:
"I could not find the answer in the uploaded documents."

Do not invent facts. Keep the answer clear and concise.

Use conversation history only to understand references
and follow-up questions. The document context remains
the only source of factual answers.

Conversation history:
{history_text}

Document context:
{context}

User question:
{query}

Answer:
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        answer = response.text or (
            "The AI model did not return a text response."
        )

    except Exception as exc:
        print(f"Gemini generation error: {exc}")

        raise HTTPException(
            status_code=502,
            detail="Answer generation failed. Check the server logs.",
        ) from exc

    sources = []

    for chunk, source_metadata in zip(
        relevant_chunks,
        relevant_metadata,
    ):
        source = {
            "filename": source_metadata.get(
                "filename",
                "Unknown document",
            ),
            "chunk_id": source_metadata.get("chunk_id"),
            "text": chunk,
        }

        if source_metadata.get("page") is not None:
            source["page"] = source_metadata["page"]

        sources.append(source)

    return {
        "question": query,
        "answer": answer,
        "sources": sources,
    }


# ==================================================
# DOCUMENT EXTRACTION
# ==================================================

def extract_document(file_path: Path):
    documents = []

    if file_path.suffix.lower() == ".txt":
        text = file_path.read_text(
            encoding="utf-8-sig"
        )

        if text.strip():
            documents.append({
                "text": text,
                "page": None,
            })

        return documents

    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(str(file_path))

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            text = page.extract_text()

            if text and text.strip():
                documents.append({
                    "text": text,
                    "page": page_number,
                })

        return documents

    return documents


# ==================================================
# TEXT CHUNKING
# ==================================================

def create_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

