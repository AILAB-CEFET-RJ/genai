# QDRANT
## Importar o client
from qdrant_client import QdrantClient
## Criar nossa coleção
from qdrant_client.http.models import Distance, VectorParams
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langchain_community.chat_models import ChatOllama

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LANGCHAIN
## Importar Qdrant como vector store
from langchain.vectorstores import Qdrant
## Importar OpenAI embeddings
# from langchain.embeddings.openai import OpenAIEmbeddings
## Função para auxiliar na quebra do texto em chunks
from langchain.text_splitter import CharacterTextSplitter
## Módulo para facilitar o uso de vector stores em QA (question answering)
from langchain.chains import RetrievalQA
## Importar LLM
# from langchain.llms import OpenAI

# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings

# PYTHON
# Variável de Ambiente
import os

client = QdrantClient(host="localhost", port=6333)

client.recreate_collection(
    collection_name="openai_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# os.environ['OPENAI_API_KEY'] = 'SUA_CHAVE_AQUI'

# embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# tokenizer = embeddings.client.tokenizer
# tokenizer.pad_token = tokenizer.eos_token

vectorstore = Qdrant(
        client=client,
        collection_name="openai_collection",
        embeddings=embeddings
    )

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

with open("../data/base_dados.txt") as f:
    raw_text = f.read()

texts = get_chunks(raw_text)

vectorstore.add_texts(texts)

MODEL_NAME = 'llama2'

llm = ChatOllama(model_name=MODEL_NAME, temperature=0)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

query = "quantos por cento das solicitações são revisadas pelo comitê?"
response = qa.run(query)

print(response)