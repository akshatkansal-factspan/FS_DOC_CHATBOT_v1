
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

# Project root: doc-chatbot/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def load_documents(folder_path=DOCUMENTS_DIR):
    """Load PDF, DOCX and TXT files into LangChain Documents."""

    folder_path = Path(folder_path)
    documents = []

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Document folder not found: {folder_path}"
        )

    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
    }

    for file_path in sorted(folder_path.rglob("*")):
        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()

        if extension not in loaders:
            continue

        try:
            loader = loaders[extension](str(file_path))
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["source"] = str(
                    file_path.relative_to(folder_path)
                )
                doc.metadata["page"] = doc.metadata.get("page")
                doc.metadata["doc_type"] = extension.lstrip(".")

            documents.extend(loaded_docs)

            print(f"Loaded: {file_path.name} ({len(loaded_docs)} documents)")

        except Exception as exc:
            print(f"ERROR loading {file_path.name}: {exc}")

    return documents


if __name__ == "__main__":
    docs = load_documents()

    print(f"\nTotal documents/pages loaded: {len(docs)}")

    if docs:
        print("\nFirst document metadata:")
        print(docs[0].metadata)

        print("\nFirst document text preview:")
        print(docs[0].page_content[:300])
