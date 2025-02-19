from qdrant_client import QdrantClient

# from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatOllama

# from langchain_community.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import logging

def main():
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_NAME = 'llama3.1'
    COLLECTION_NAME = "demo_collection"
    QUERY = "quantos por cento das solicitações são revisadas pelo comitê?"

    llm = ChatOllama(model_name=MODEL_NAME, 
                     temperature=0)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    client = QdrantClient(host="localhost", port=6333)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=vector_store.as_retriever()
    # )

    # response = qa.invoke(QUERY)

    # print(f'Query:\n{response['query']}\nResult:\n{response['result']}')

    # print('<<<2nd>>>')
    system_prompt = (
        "Use o contexto fornecido para responder ao que se pede. "
        "Se você não souber a resposta, responda que não sabe. "
        "Use no máximo três frases na sua resposta e mantenha essa resposta concisa. "
        "Responda sempre usando Português."
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(vector_store.as_retriever(), question_answer_chain)

    response = chain.invoke({"input": QUERY})
    print(response['answer'])


if __name__=="__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s',
        datefmt='%H:%M:%S'
    )

    logging.info("### Starting up")
    main()