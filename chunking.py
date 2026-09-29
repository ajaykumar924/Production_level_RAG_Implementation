from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_ingestion import load_document

def chunk_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Step 2: Document Chunking
    Splits loaded documents into smaller, overlapping chunks using RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"[Chunking] Created {len(chunks)} chunks from {len(documents)} pages.")
    return chunks

if __name__ == "__main__":
    docs = load_document()
    chunks = chunk_documents(docs)
    print(f"\nSample Chunk 1 (length {len(chunks[0].page_content)}):\n{chunks[0].page_content[:300]}...")
