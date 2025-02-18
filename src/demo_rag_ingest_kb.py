from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# from langchain_community.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore

from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings

import logging

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
    ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
    COLLECTION_NAME = "demo_collection"

    client = QdrantClient(host="localhost", port=6333)

    # client.recreate_collection(
    #     collection_name="demo_collection",
    #     vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    # )
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
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

    with open("../data/base_dados.txt") as f:
        raw_text = f.read()

    texts = get_chunks(raw_text)

    vectorstore.add_texts(texts)

if __name__=="__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.info("### Starting up")
    main()