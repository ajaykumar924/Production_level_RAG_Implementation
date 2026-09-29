import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from document_ingestion import load_document
from chunking import chunk_documents
from vector_embedding import load_vector_store, create_vector_store, PERSIST_DIRECTORY

# Load environment variables (.env)
load_dotenv()
groq_api_key = os.getenv("Groq_API_key") or os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="HR Policy RAG Assistant", page_icon="📄", layout="centered")

st.title("📄 HR Policy Manual - RAG Assistant")
st.caption("A simple RAG pipeline using LangChain, HuggingFace MiniLM, ChromaDB, and Groq (qwen/qwen3.8-27b).")

# Cache vector store to load the model into memory only once
@st.cache_resource
def get_cached_vector_store():
    return load_vector_store(PERSIST_DIRECTORY)

vector_store = get_cached_vector_store()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("**LLM:** Groq (`qwen/qwen3.8-27b`)\n\n**Embeddings:** HuggingFace `MiniLM`\n\n**Vector DB:** ChromaDB")
    
    if st.button("🔄 (Re)Index Document"):
        with st.spinner("Ingesting, chunking, and embedding PDF into ChromaDB..."):
            docs = load_document()
            chunks = chunk_documents(docs)
            create_vector_store(chunks)
            st.cache_resource.clear()
            st.success("Indexing complete! Vector store updated.")
            st.rerun()

if not vector_store:
    st.warning("⚠️ Vector database not found. Click below to index the HR Policy Manual.")
    if st.button("🚀 Build Vector Store Now"):
        with st.spinner("Reading 'data/HR Policy Manual.pdf' and generating embeddings..."):
            docs = load_document()
            chunks = chunk_documents(docs)
            create_vector_store(chunks)
            st.cache_resource.clear()
            st.success("Vector store created! You can now ask questions below.")
            st.rerun()
else:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("🔍 View Retrieved Context Chunks (RAG Source)"):
                    for i, (source_text, page) in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk {i} (Page {page}):**")
                        st.write(source_text)

    # Chat Input
    user_query = st.chat_input("Ask a question about HR policies (e.g., What is the leave policy?)...")

    if user_query:
        if not groq_api_key:
            st.error("Groq API key not found. Please verify `Groq_API_key` in your `.env` file.")
        else:
            # Display user message
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            # Generate RAG response
            with st.chat_message("assistant"):
                with st.spinner("Retrieving context and generating answer..."):
                    # 1. RETRIEVE: similarity search in ChromaDB
                    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                    relevant_docs = retriever.invoke(user_query)
                    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

                    # 2. AUGMENT: Inject retrieved context into prompt
                    rag_prompt = f"""You are a helpful HR Assistant. Answer the question based ONLY on the provided HR Policy Manual context below.
If the answer is not mentioned in the context, say: "I could not find that information in the HR Policy Manual."

Context:
{context_text}

Question: {user_query}
Answer:"""

                    # 3. GENERATE: Call Groq Qwen LLM
                    llm = ChatGroq(
                        model="qwen/qwen3.8-27b",
                        groq_api_key=groq_api_key,
                        temperature=0.2
                    )
                    response = llm.invoke(rag_prompt)
                    answer = response.content
                    st.write(answer)

                    # Save sources for history display
                    sources = [(d.page_content, d.metadata.get("page", "N/A")) for d in relevant_docs]
                    with st.expander("🔍 View Retrieved Context Chunks (RAG Source)"):
                        for i, (source_text, page) in enumerate(sources, 1):
                            st.markdown(f"**Chunk {i} (Page {page}):**")
                            st.write(source_text)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
