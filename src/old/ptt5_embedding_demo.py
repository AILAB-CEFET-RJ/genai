from transformers import AutoTokenizer, AutoModel
import torch

def get_sentence_embedding(sentence, model, tokenizer):
    """Gera o embedding de uma sentença usando."""
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.encoder(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Carregar o modelo e o tokenizador
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Exemplo de frases em português
sentences = [
    "O clima hoje está ensolarado e agradável.",
    "Estou estudando aprendizado de máquina com transformers.",
    "O futebol é o esporte mais popular do Brasil."
]

# Gerar e exibir embeddings
for sentence in sentences:
    embedding = get_sentence_embedding(sentence, model, tokenizer)
    print(f"Embedding para: '{sentence}'\n{embedding[:5]}... (mostrando primeiros 5 valores)\n")
