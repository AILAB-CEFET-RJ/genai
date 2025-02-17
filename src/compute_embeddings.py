import fitz  # PyMuPDF
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import sys

# Configurations
# PDF_FILE = "../data/tokio_outubro_2024.pdf"  # Change this to your PDF file
PDF_FILE = "../data/teste.pdf"  # Change this to your PDF file
QDRANT_LOCATION = "./qdrant_db"  # Persistent storage
COLLECTION_NAME = "pdf_chunks"
CHUNK_SIZE = 500  # Adjust as needed
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text


def chunk_text(text, chunk_size, chunk_overlap):
    """Splits text into chunks using LangChain's RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

import unicodedata

def clean_text(text):
    """Normalize and clean text to avoid encoding issues."""
    text = text.replace("\x00", "")  # Remove null characters
    text = text.replace("\x2e", "")  # Remove null characters
    text = unicodedata.normalize("NFKC", text)  # Normalize Unicode
    text = text.encode("utf-8", "ignore").decode("utf-8")  # Ignore problematic characters
    return text.strip()

# def compute_embeddings(texts, model_name):
#     """Computes embeddings for a list of cleaned texts."""
#     model = SentenceTransformer(model_name)
#     cleaned_texts = [clean_text(t) for t in texts]  # Clean each text chunk
#     return model.encode(cleaned_texts, batch_size=16, show_progress_bar=True)

# def compute_embeddings(texts, model_name):
#     """Computes embeddings for a list of texts."""
#     model = SentenceTransformer(model_name)
#     return model.encode(texts, batch_size=16, show_progress_bar=True)

def compute_embeddings(texts, model_name):
    """Computes embeddings for a list of texts and verifies output."""
    model = SentenceTransformer(model_name)
    cleaned_texts = [t.strip() for t in texts]  # Strip whitespace

    # Debug: Print first 5 text chunks
    print("🔍 Sample text chunks before embedding:")
    for i, text in enumerate(cleaned_texts[:5]):
        print(f"Chunk {i+1}: {text[:100]}...")  # Print first 100 chars

    embeddings = model.encode(cleaned_texts, batch_size=16, show_progress_bar=True)

    # Debug: Check if embeddings are correct
    print("✅ Successfully computed embeddings!")
    print(f"Embedding shape: {len(embeddings)}, {len(embeddings[0])} (First vector)")

    return embeddings

def store_in_qdrant(embeddings, texts, location, collection_name):
    """Stores text chunks and embeddings in Qdrant."""
    client = QdrantClient(location=location)
    client.recreate_collection(collection_name=collection_name, vectors_config={"size": len(embeddings[0]), "distance": "Cosine"})
    
    points = [
        PointStruct(id=int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**9),
                    vector=embeddings[i],
                    payload={"text": texts[i]})
        for i in range(len(texts))
    ]
    
    client.upsert(collection_name=collection_name, points=points)
    print("✅ Successfully stored chunks in Qdrant!")


if __name__ == "__main__":
    print("📄 Extracting text from PDF...")
    text = extract_text_from_pdf(PDF_FILE)
    
    print("✂️ Splitting text into chunks...")
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    
    print("🧠 Computing embeddings...")
    embeddings = compute_embeddings(chunks, EMBEDDING_MODEL)
    
    print("📥 Storing embeddings in Qdrant...")
    store_in_qdrant(embeddings, chunks, QDRANT_LOCATION, COLLECTION_NAME)
    
    print("🎉 All done!")