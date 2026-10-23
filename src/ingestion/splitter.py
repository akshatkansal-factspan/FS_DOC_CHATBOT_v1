
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.loader import load_documents


def split_documents(documents):
    """Split loaded documents into chunks while preserving metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    import random

    documents = load_documents()
    chunks = split_documents(documents)

    # Basic statistics
    unique_sources = {
        doc.metadata.get("source")
        for doc in documents
    }

    page_count = sum(
        1 for doc in documents
        if doc.metadata.get("page") is not None
    )

    average_length = (
        sum(len(chunk.page_content) for chunk in chunks)
        / len(chunks)
        if chunks else 0
    )

    print("\n--- INGESTION STATISTICS ---")
    print(f"Unique files: {len(unique_sources)}")
    print(f"Documents/pages loaded: {len(documents)}")
    print(f"Documents with page metadata: {page_count}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Average chunk length: {average_length:.2f} characters")

    # Inspect up to 10 random chunks
    sample_size = min(10, len(chunks))
    sampled_chunks = random.sample(chunks, sample_size)

    print(f"\n--- RANDOM CHUNK SAMPLES ({sample_size}) ---")

    for i, chunk in enumerate(sampled_chunks, start=1):
        print(f"\nChunk {i}")
        print(f"Metadata: {chunk.metadata}")
        print(f"Length: {len(chunk.page_content)} characters")
        print(f"Text preview: {chunk.page_content[:250]}")
