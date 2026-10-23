
import os
from pathlib import Path

from dotenv import load_dotenv

# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load settings from the project's .env file
load_dotenv(PROJECT_ROOT / ".env")

# Model settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv(
    "OLLAMA_EMBED_MODEL",
    "nomic-embed-text",
)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Data and storage paths
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

# Text splitting and retrieval settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 2
