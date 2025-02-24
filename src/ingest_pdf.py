import argparse
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# from langchain_community.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings

import logging
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process a PDF and store embeddings in Qdrant.")
    parser.add_argument("collection_name", type=str, help="Name of the Qdrant collection.")
    parser.add_argument("pdf_file", type=str, help="Path to the PDF file.")
    parser.add_argument("extraction_strategy", type=str, help="Strategy to extract content ('SIMPLE' or 'STRUCTURED').")
    args = parser.parse_args()

    # ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
    ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    COLLECTION_NAME = args.collection_name
    PDF_FILE = args.pdf_file
    EXTRACTION_STRATEGY = args.extraction_strategy

    client = QdrantClient(host="localhost", port=6333)

    # client.recreate_collection(
    #     collection_name="demo_collection",
    #     vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    # )
    if True or not client.collection_exists(COLLECTION_NAME):
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    embeddings = HuggingFaceEmbeddings(model_name=ENCODER)

    vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings
        )

    print("📄 Extracting text from PDF...")
    document_raw_text = extract_text_from_pdf(PDF_FILE)
    if EXTRACTION_STRATEGY == "SIMPLE":
        texts = get_chunks(document_raw_text)
        print(len(texts))
        vector_store.add_texts(texts)
    elif EXTRACTION_STRATEGY == "STRUCTURED":
        uuids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=uuids)

# python ingest_pdf.py TokioMarine_collection ../data/CG-Tokio-Marine-2025.pdf
# python ingest_pdf.py PortoSeguro_collection ../data/CG-Porto-Seguro-2025.pdf
if __name__=="__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.info("### Starting up")
    main()