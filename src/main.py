from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
# set up Ollama Embeddings: https://python.langchain.com/docs/integrations/text_embedding/ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

print('>>> Creating PyPDFLoader...')
loader = PyPDFLoader('../data/tokio_outubro_2024.pdf')

print('>>> Loading document\'s pages...')
pages = loader.load()

# define the text splitter
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200, 
    separators=["\n\n", "\n", " ", ""]
)

print('>>> Splitting document into chuncks...')

# Create our splits from the PDF
docs = r_splitter.split_documents(pages)

print('>>> Creating embeddings...')
MODEL_NAME = 'llama2'
embeddings = OllamaEmbeddings(model=MODEL_NAME) 

print('>>> Storing embeddings in Qdrant...')
# set up the qdrant database
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="my_documents",
)

print('>>> Creating ChatOllama...')

# model name can be any model you have installed with Ollama
# complete list of models available @ Ollama: https://ollama.ai/library
llm = ChatOllama(model_name=MODEL_NAME, temperature=0)

question = '''
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
result = llm({"query": question})
# from langchain.chains.retrieval_qa.base import RetrievalQA
# qa_chain_mr = RetrievalQA.from_chain_type(
#     llm,
#     retriever=qdrant.as_retriever(),
#     chain_type="map_reduce"
# )
# result = qa_chain_mr({"query": question})

print(result["result"])