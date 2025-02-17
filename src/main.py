from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
# set up Ollama Embeddings: https://python.langchain.com/docs/integrations/text_embedding/ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains.retrieval_qa.base import RetrievalQA
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings

# print('>>> Creating PyPDFLoader...')
# loader = PyPDFLoader('../data/tokio_outubro_2024.pdf')

# print('>>> Loading document\'s pages...')
# pages = loader.load()

# # define the text splitter
# r_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1500,
#     chunk_overlap=200, 
#     separators=["\n\n", "\n", " ", ""]
# )

# print('>>> Splitting document into chuncks...')

# # Create our splits from the PDF
# docs = r_splitter.split_documents(pages)

# print('>>> Creating embeddings...')
MODEL_NAME = 'llama2'
# embeddings = OllamaEmbeddings(model=MODEL_NAME) 

# location="./qdrant_db"

# print('>>> Storing embeddings in Qdrant...')
# # set up the qdrant database
# qdrant = Qdrant.from_documents(
#     docs,
#     embeddings,
#     # location=":memory:",  # Local mode with in-memory storage only
#     location=location,  # Local mode with disk storage
#     collection_name="SegurIA",
# )

print('>>> Creating ChatOllama...')

# model name can be any model you have installed with Ollama
# complete list of models available @ Ollama: https://ollama.ai/library
llm = ChatOllama(model_name=MODEL_NAME, temperature=0)

QUERY = '''
    Considere a contratação de um seguro de condomínio, com várias coberturas, feita por um condomínio vertical residencial, constituído por nove blocos. 
    Dentre as coberturas contratadas, está a cobertura básica ampla. Este seguro, com todas as coberturas contratadas, foi objeto de renovação com a 
    mesma seguradora.
    - No caso desmoronamento parcial na cobertura e nos andares imediatamente abaixo, haveria cobertura do seguro? Justifique.
    - No caso de ter sido realizada obra na cobertura que teve desabamento parcial, dentro das regras do condomínio e da 
        prefeitura municipal, haveria cobertura? Justifique.
    - No caso de incêndio em carro elétrico ocorrido na garagem subterrânea do condomínio, com incêndio e explosão, 
        caso haja comprometimento da laje, haverá cobertura do seguro? Justifique.
    '''

print('>>> Retrieving answer...')

# The embedding model that will be used by the collection
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # You can replace with any other HuggingFace model

# Initialize Qdrant client
qdrant_client = QdrantClient(url="http://localhost:6333")

# Initialize Qdrant retriever
qdrant = Qdrant(qdrant_client, 
                   embeddings=embeddings, 
                   collection_name="SegurIA")

retriever = qdrant.as_retriever()

print(retriever)
print("####")

# qa_chain_mr = RetrievalQA.from_chain_type(
#     llm,
#     retriever=retriever,
#     chain_type="map_reduce"
# )
# result = qa_chain_mr({"query": question})

# print(result["result"])


qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# QUERY = "WRITE QUERY RELATED TO AN ARTICLE YOU ADDED"

answer = qa.run(QUERY)
print(answer)