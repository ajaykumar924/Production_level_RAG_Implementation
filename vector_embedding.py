import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from document_ingestion import load_document
from chunking import chunk_documents

load_dotenv()

# Embedding model: MiniLM from HuggingFace
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "chroma_db")

def get_embedding_model():
    """Initializes and returns the HuggingFace MiniLM embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def create_vector_store(chunks, persist_dir: str = PERSIST_DIRECTORY):
    """
    Step 3: Vector Embedding & Storage
    Converts chunks into vector embeddings and stores them in ChromaDB.
    """
    embeddings = get_embedding_model()
    print("[Vector Embedding] Indexing chunks into ChromaDB (this may take a minute)...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"[Vector Embedding] Vector store saved successfully at '{persist_dir}'.")
    return vector_store

def load_vector_store(persist_dir: str = PERSIST_DIRECTORY):
    """Loads an existing Chroma vector store from the local directory."""
    if not os.path.exists(persist_dir):
        return None
    embeddings = get_embedding_model()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

if __name__ == "__main__":
    # 1. Ingest
    docs = load_document()
    # 2. Chunk
    chunks = chunk_documents(docs)
    # 3. Embed & Store
    create_vector_store(chunks)
