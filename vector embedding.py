# Re-exporting from vector_embedding.py
# (Note: In Python, standard module names use underscores: 'vector_embedding')
from vector_embedding import (
    get_embedding_model,
    create_vector_store,
    load_vector_store,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY
)

if __name__ == "__main__":
    from document_ingestion import load_document
    from chunking import chunk_documents

    docs = load_document()
    chunks = chunk_documents(docs)
    create_vector_store(chunks)
