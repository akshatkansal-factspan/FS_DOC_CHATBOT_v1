import streamlit as st
import uuid

from src.graph.builder import build_graph
from src.ingestion.ingest import ingest_documents
from src.ingestion.google_drive import ingest_google_drive_folder
from src.auth.google_drive_oauth import (
    get_google_drive_credentials,
    process_google_oauth_callback,
    start_google_oauth,
)
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
# 2. Handle Google OAuth callback and session state
# --------------------------------------------------

try:
    process_google_oauth_callback()
except Exception as error:
    st.error(f"Google sign-in failed: {error}")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------------------------
# 3. Google Drive folder import
# --------------------------------------------------

with st.expander("📁 Import documents from Google Drive", expanded=False):
    credentials = get_google_drive_credentials()

    if credentials:
        st.success("Google Drive connected.")
        if st.button("Disconnect Google Drive"):
            st.session_state.pop("google_drive_credentials", None)
            st.rerun()

        folder_url = st.text_input(
            "Google Drive folder URL",
            placeholder="https://drive.google.com/drive/folders/...",
            help="The signed-in Google account must have access to this folder.",
            key="google_drive_folder_url",
        )

        if st.button("Ingest Drive folder", type="primary"):
            if not folder_url.strip():
                st.warning("Enter a Google Drive folder URL first.")
            else:
                with st.spinner("Reading files from Google Drive and updating the index..."):
                    try:
                        summary = ingest_google_drive_folder(folder_url, credentials)
                        st.success(
                            f"Drive import complete. Loaded {summary['files']} supported "
                            f"files, created {summary['chunks']} chunks, and updated the index."
                        )
                        if summary["skipped"]:
                            st.info(
                                "Skipped unsupported Google Drive items: "
                                + ", ".join(summary["skipped"][:10])
                                + (" ..." if len(summary["skipped"]) > 10 else "")
                            )
                        st.rerun()
                    except Exception as error:
                        st.error(f"Google Drive ingestion failed: {error}")
    else:
        st.write(
            "Sign in with the Google account that can access the folder. "
            "Only supported document types are imported."
        )
        if st.button("Sign in with Google Drive", type="primary"):
            try:
                auth_url = start_google_oauth()
                st.link_button("Continue to Google", auth_url, use_container_width=True)#test
                st.info(
                    "Select **Continue to Google** above. After consent, Google will "
                    "return you to this app."
                )
            except Exception as error:
                st.error(f"Could not start Google sign-in: {error}")


# --------------------------------------------------
# 4. Sidebar controls
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

    if st.button("Re-ingest local documents", use_container_width=True):
        with st.spinner("Ingesting local documents..."):
            try:
                ingest_documents()
                st.success("Local documents ingested successfully!")
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
# 5. Display existing conversation and sources
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
# 6. Accept a question and run the graph
# --------------------------------------------------

question = st.chat_input("Ask a question about your documents...")

if question:
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

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                result = st.session_state.graph.invoke(initial_state, config=config)
                answer = result.get("generation", "I couldn't generate an answer.")
            except Exception as error:
                answer = f"An error occurred: {error}"

        st.markdown(answer)

        if result:
            for doc in result.get("documents", []):
                filename = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page")
                label = f"{filename}, page {page + 1}" if isinstance(page, int) else filename
                excerpt = doc.page_content[:400]
                if len(doc.page_content) > 400:
                    excerpt += "..."
                source_details.append({"label": label, "excerpt": excerpt})

        if source_details:
            with st.expander(f"Sources ({len(source_details)})"):
                for i, source in enumerate(source_details, start=1):
                    st.markdown(f"**{i}. {source['label']}**")
                    st.caption(source["excerpt"])

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": source_details,
    })
