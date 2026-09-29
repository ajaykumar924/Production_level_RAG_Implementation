import os
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader

# Default path to the HR Policy Manual
DEFAULT_PDF_PATH = os.path.join(os.path.dirname(__file__), "data", "HR Policy Manual.pdf")

def load_document(file_path: str = DEFAULT_PDF_PATH):
    """
    Step 1: Document Ingestion
    Loads a PDF document and returns a list of LangChain Document objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Use PyMuPDFLoader (fast); fallback to PyPDFLoader if needed
    try:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
    except Exception:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

    print(f"[Document Ingestion] Successfully loaded {len(documents)} pages from '{file_path}'.")
    return documents

if __name__ == "__main__":
    docs = load_document()
    print(f"Sample content from Page 1:\n{docs[0].page_content[:300]}...")
