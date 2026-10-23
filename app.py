
import streamlit as st
import uuid

from src.graph.builder import build_graph
from src.ingestion.ingest import ingest_documents
from src.retrieval.retriever import load_vectorstore


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Document Chatbot",
    page_icon=":robot:",
    layout="wide",
)

st.title("📄 GenAI Document Chatbot")
st.caption("Ask questions about your uploaded documents.")


# --------------------------------------------------
# 2. Initialize session state
# --------------------------------------------------

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------------------------
# 3. Sidebar controls
# --------------------------------------------------

with st.sidebar:
    st.header("Document Settings")
    retrieval_k = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=6,
        value=4,
        help="Choose how many document chunks to search for each question.",
    )

    if st.button("Re-ingest documents", use_container_width=True):
        with st.spinner("Ingesting documents..."):
            try:
                ingest_documents()
                st.success("Documents ingested successfully!")
                st.rerun()
            except Exception as error:
                st.error(f"Ingestion failed: {error}")

    st.divider()
    st.subheader("Index Status")

    try:
        vectorstore = load_vectorstore()
        chunk_count = vectorstore.index.ntotal
        st.metric("Indexed chunks", chunk_count)
    except Exception as error:
        st.warning(f"Could not load index: {error}")

    st.divider()

    if st.button("New Chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()


# --------------------------------------------------
# 4. Display existing conversation and sources
# --------------------------------------------------

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        sources = message.get("sources", [])

        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for i, source in enumerate(sources, start=1):
                    st.markdown(f"**{i}. {source['label']}**")
                    st.caption(source["excerpt"])


# --------------------------------------------------
# 5. Accept a question and run the graph
# --------------------------------------------------

question = st.chat_input("Ask a question about your documents...")

if question:

    # Display and save the user's question
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    initial_state = {
        "messages": [],
        "question": question,
        "rewritten_question": "",
        "documents": [],
        "generation": "",
        "retry_count": 0,
        "route": "",
        "retrieval_k": retrieval_k,
    }

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    result = None
    source_details = []

    # Run the existing LangGraph workflow
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                result = st.session_state.graph.invoke(
                    initial_state,
                    config=config,
                )

                answer = result.get(
                    "generation",
                    "I couldn't generate an answer.",
                )

            except Exception as error:
                answer = f"An error occurred: {error}"

        st.markdown(answer)

        # Extract sources from the graph's returned documents
        if result:
            for doc in result.get("documents", []):
                filename = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page")

                # PDF loaders often store pages as zero-based numbers
                if isinstance(page, int):
                    label = f"{filename}, page {page + 1}"
                else:
                    label = filename

                excerpt = doc.page_content[:400]

                if len(doc.page_content) > 400:
                    excerpt += "..."

                source_details.append({
                    "label": label,
                    "excerpt": excerpt,
                })

        # Display sources under the answer
        if source_details:
            with st.expander(f"Sources ({len(source_details)})"):
                for i, source in enumerate(source_details, start=1):
                    st.markdown(f"**{i}. {source['label']}**")
                    st.caption(source["excerpt"])

    # Save answer and source information for future reruns
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": source_details,
    })
