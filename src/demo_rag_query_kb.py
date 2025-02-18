from qdrant_client import QdrantClient

# from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatOllama

# from langchain_community.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore

from langchain.chains import RetrievalQA
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings

import logging

def main():
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_NAME = 'llama3.1'
    COLLECTION_NAME = "demo_collection"

    llm = ChatOllama(model_name=MODEL_NAME, 
                     temperature=0)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    client = QdrantClient(host="localhost", port=6333)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    query = "quantos por cento das solicitações são revisadas pelo comitê?"
    response = qa.invoke(query)

    print(f'Query:\n{response['query']}\nResult:\n{response['result']}')

if __name__=="__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s',
        datefmt='%H:%M:%S'
    )

    logging.info("### Starting up")
    main()