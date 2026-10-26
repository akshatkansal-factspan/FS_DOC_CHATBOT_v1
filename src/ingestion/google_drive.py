import io
import re
import tempfile
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import VECTORSTORE_DIR
from src.ingestion.loader import load_documents
from src.ingestion.splitter import split_documents
from src.retrieval.retriever import load_vectorstore


FOLDER_MIME = "application/vnd.google-apps.folder"
GOOGLE_DOC_EXPORTS = {
    "application/vnd.google-apps.document": (".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    "application/vnd.google-apps.spreadsheet": (".csv", "text/csv"),
    "application/vnd.google-apps.presentation": (".pdf", "application/pdf"),
}
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}


def _folder_id_from_url(folder_url):
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", folder_url)
    if match:
        return match.group(1)
    # Also allow a bare folder ID for convenience.
    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", folder_url.strip()):
        return folder_url.strip()
    raise ValueError("That doesn't look like a Google Drive folder URL.")


def _list_folder_items(service, folder_id):
    items = []
    page_token = None
    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            pageSize=1000,
            pageToken=page_token,
            fields="nextPageToken, files(id,name,mimeType,webViewLink)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        items.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return items


def _download_drive_file(service, item, destination):
    mime = item["mimeType"]
    file_id = item["id"]
    name = item["name"]

    if mime in GOOGLE_DOC_EXPORTS:
        extension, export_mime = GOOGLE_DOC_EXPORTS[mime]
        if not name.lower().endswith(extension):
            name = name + extension
        request = service.files().export_media(fileId=file_id, mimeType=export_mime)
    elif mime.startswith("application/vnd.google-apps."):
        return None
    else:
        extension = Path(name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            return None
        request = service.files().get_media(
            fileId=file_id,
            supportsAllDrives=True,
        )

    # Prefix the Drive ID so same-named files don't overwrite one another.
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "document"
    local_path = destination / f"{file_id}_{safe_name}"
    with io.FileIO(local_path, "wb") as output:
        downloader = MediaIoBaseDownload(output, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return local_path


def _collect_files_recursive(service, folder_id, destination, seen_folders=None):
    if seen_folders is None:
        seen_folders = set()
    if folder_id in seen_folders:
        return [], []
    seen_folders.add(folder_id)

    downloaded = []
    skipped = []
    for item in _list_folder_items(service, folder_id):
        if item["mimeType"] == FOLDER_MIME:
            nested_files, nested_skipped = _collect_files_recursive(
                service, item["id"], destination, seen_folders
            )
            downloaded.extend(nested_files)
            skipped.extend(nested_skipped)
            continue

        path = _download_drive_file(service, item, destination)
        if path is None:
            skipped.append(item["name"])
        else:
            downloaded.append((path, item["id"], item["name"], folder_id))
    return downloaded, skipped


def ingest_google_drive_folder(folder_url, credentials):
    """Download supported files from a Drive folder and append/update their chunks in FAISS."""
    folder_id = _folder_id_from_url(folder_url)
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)

    with tempfile.TemporaryDirectory(prefix="drive_import_") as temp_dir:
        temp_path = Path(temp_dir)
        downloaded, skipped = _collect_files_recursive(service, folder_id, temp_path)

        if not downloaded:
            raise ValueError(
                "No supported files were found. Supported formats: PDF, DOCX, TXT, CSV, "
                "Google Docs, Google Sheets (CSV export), and Google Slides (PDF export)."
            )

        documents = load_documents(temp_path)
        id_by_source = {}
        for local_path, file_id, original_name, containing_folder_id in downloaded:
            relative_source = str(local_path.relative_to(temp_path))
            id_by_source[relative_source] = (file_id, original_name, folder_id)

        for doc in documents:
            source = doc.metadata.get("source", "")
            file_info = id_by_source.get(source)
            if file_info:
                file_id, original_name, root_folder_id = file_info
                doc.metadata["source"] = original_name
                doc.metadata["drive_file_id"] = file_id
                doc.metadata["drive_folder_id"] = root_folder_id
                doc.metadata["source_type"] = "google_drive"

        chunks = split_documents(documents)
        if not chunks:
            raise ValueError("Supported files were downloaded, but no text chunks could be extracted.")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Keep existing local documents, but replace prior chunks from this same Drive folder.
        
        # Load the existing index and keep the same embedding model everywhere.
        try:
            vectorstore = load_vectorstore()

            existing_docs = list(vectorstore.docstore._dict.values())

            # Remove old chunks from this folder in memory.
            remaining_docs = [
                doc for doc in existing_docs
                if doc.metadata.get("drive_folder_id") != folder_id
            ]

            # Detect a legacy index built with a different embedding dimension.
            new_dimension = len(embeddings.embed_query("dimension check"))
            old_dimension = vectorstore.index.d

            if old_dimension != new_dimension:
                # Rebuild using all existing documents except this folder,
                # then add the newly ingested Drive chunks.
                all_documents = remaining_docs + chunks
                vectorstore = FAISS.from_documents(
                    documents=all_documents,
                    embedding=embeddings,
                )
            else:
                existing_ids = [
                    doc_id
                    for doc_id, doc in vectorstore.docstore._dict.items()
                    if doc.metadata.get("drive_folder_id") == folder_id
                ]

                if existing_ids:
                    vectorstore.delete(ids=existing_ids)

                vectorstore.add_documents(chunks)

        except (FileNotFoundError, RuntimeError, ValueError):
            # If no usable index exists, rebuild with local documents too.
            local_documents = load_documents()
            local_chunks = split_documents(local_documents)

            # Avoid duplicating the Drive chunks in the rebuild.
            all_documents = local_chunks + chunks

            vectorstore = FAISS.from_documents(
                documents=all_documents,
                embedding=embeddings,
            )

        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(VECTORSTORE_DIR))

    return {
        "files": len(downloaded),
        "chunks": len(chunks),
        "skipped": skipped,
    }
