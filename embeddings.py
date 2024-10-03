
from langchain_community.embeddings import OllamaEmbeddings

def generate_embeddings():
    """Generate embeddings using the OllamaEmbeddings model"""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
