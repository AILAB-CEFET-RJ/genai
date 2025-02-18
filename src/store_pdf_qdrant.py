import pdfplumber

# from llama_index.node_parser import SimpleNodeParser
from llama_index.core.node_parser import SimpleNodeParser

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import hashlib
from llama_index.core import Document, VectorStoreIndex

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    print()
    print(text)
    print()
    return Document(text)

# Function to generate a unique ID for each text chunk
def generate_id(text):
    return hashlib.md5(text.encode()).hexdigest()

# Qdrant setup
QDRANT_HOST = "localhost"  # Change to the actual Qdrant host if needed
QDRANT_PORT = 6333
COLLECTION_NAME = "pdf_embeddings"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Create collection if not exists
if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),  # Adjust size to match embedding model
    )

# Load embedding model
# embed_model = HuggingFaceEmbedding(model_name="hkunlp/instructor-xl")
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

# Load PDF and extract text
pdf_path = "../data/tokio_outubro_2024.pdf"  # Change to your PDF file path
text = extract_text_from_pdf(pdf_path)

# Split text into chunks
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents([text])

# Convert text chunks to embeddings and store in Qdrant
data_points = []
for node in nodes:
    chunk_text = node.get_text()
    embedding = embed_model.get_text_embedding(chunk_text)
    data_points.append(PointStruct(id=generate_id(chunk_text), vector=embedding, payload={"text": chunk_text}))

# Upsert embeddings to Qdrant
try:
    client.upsert(collection_name=COLLECTION_NAME, points=data_points)
    print("PDF stored successfully in Qdrant!")
except Exception as e:
    print(f"Error storing PDF: {e}")
