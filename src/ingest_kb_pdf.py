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
    # ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
    ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    COLLECTION_NAME = "SegurIA_collection"
    PDF_FILE = "../data/tokio_outubro_2024.pdf"  # Change this to your PDF file

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

    # os.environ['OPENAI_API_KEY'] = 'TBD'
    # embeddings = OpenAIEmbeddings()

    embeddings = HuggingFaceEmbeddings(model_name=ENCODER)

    vectorstore = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings
        )

    print("📄 Extracting text from PDF...")
    raw_text = extract_text_from_pdf(PDF_FILE)
    # with open("../data/base_dados.txt") as f:
    #     raw_text = f.read()

    texts = get_chunks(raw_text)

    print()
    print(len(texts))

    vectorstore.add_texts(texts)

if __name__=="__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.info("### Starting up")
    main()