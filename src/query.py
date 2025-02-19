import argparse
import logging
import pprint

from qdrant_client import QdrantClient
from langchain_community.chat_models import ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def parse_args():
    parser = argparse.ArgumentParser(description="Run a query on Qdrant with a specified LLM and embedding model.")
    parser.add_argument("--embedding_model", type=str, required=True, help="The embedding model to use.")
    parser.add_argument("--model_name", type=str, required=True, help="The language model to use.")
    parser.add_argument("--collection_name", type=str, required=True, help="The name of the Qdrant collection.")
    parser.add_argument("--query", type=str, required=True, help="The query to execute.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    llm = ChatOllama(model_name=args.model_name, temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)
    
    client = QdrantClient(host="localhost", port=6333)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=args.collection_name,
        embedding=embeddings
    )
    
    system_prompt = (
        "Use o contexto fornecido para responder ao que se pede. "
        "Se você não souber a resposta, responda que não sabe. "
        "Use no máximo três frases na sua resposta."
        "Gere uma resposta concisa. "
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
    
    response = chain.invoke({"input": args.query})
    print('Answer:')
    pprint.pp(response['answer'])
    print('Context:')
    for ctx in response['context']:
        pprint.pp(ctx)
        print()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.info("### Starting up")
    main()

# python query.py --embedding_model "sentence-transformers/all-MiniLM-L6-v2" --model_name "llama3.1" --collection_name "demo_collection" --query "quantos por cento das solicitações são revisadas pelo comitê?"
# python query.py --embedding_model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" --model_name "llama3.1" --collection_name "SegurIA_collection" --query "Quais riscos não são cobertos em danos elétricos?"
