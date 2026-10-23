
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from src.config import OLLAMA_EMBED_MODEL, VECTORSTORE_DIR, RETRIEVAL_K


def load_vectorstore():
    """Load the persisted FAISS vector store."""
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_retriever(k: int | None = None):
    """Create a retriever using the requested number of chunks."""
    vectorstore = load_vectorstore()

    # Use the configured default when k isn't provided.
    if k is None:
        k = RETRIEVAL_K

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": max(k * 4, 20),
            "lambda_mult": 0.5,
        },
    )


if __name__ == "__main__":
    retriever = get_retriever()

    question = "How many days of annual paid leave do employees receive?"
    results = retriever.invoke(question)

    print(f"Question: {question}\n")

    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Page: {doc.metadata.get('page', 'N/A')}")
        print(doc.page_content[:500])
        print()
