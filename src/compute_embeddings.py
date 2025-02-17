import pickle
from tqdm import tqdm

# Compute and cache embeddings
embedding_model = SomeEmbeddingModel()  # Example: OpenAIEmbeddings(), HuggingFaceEmbeddings()
doc_texts = [doc.page_content for doc in docs]  # Extract text from LangChain Document objects

print("Computing embeddings...")
embeddings = [embedding_model.embed(text) for text in tqdm(doc_texts)]

# Save to disk (optional)
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)
