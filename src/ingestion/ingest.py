
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import VECTORSTORE_DIR
from src.ingestion.loader import load_documents
from src.ingestion.splitter import split_documents


def ingest_documents():
    print("Loading documents...", flush=True)
    documents = load_documents()

    if not documents:
        print("No supported documents found.", flush=True)
        return

    print(f"Loaded {len(documents)} documents/pages.", flush=True)

    print("Splitting documents into chunks...", flush=True)
    chunks = split_documents(documents)

    if not chunks:
        print("No chunks were created.", flush=True)
        return

    print(f"Created {len(chunks)} chunks.", flush=True)

    print("Initializing Hugging Face embeddings...", flush=True)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating FAISS vector store...", flush=True)
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    print("Saving FAISS index...", flush=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    print("\nIngestion complete!", flush=True)
    print(f"Stored {len(chunks)} chunks.", flush=True)
    print(f"Vector database location: {VECTORSTORE_DIR}", flush=True)


if __name__ == "__main__":
    ingest_documents()
