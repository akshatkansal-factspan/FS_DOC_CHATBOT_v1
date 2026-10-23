
# GenAI Document Chatbot

A local document question-answering chatbot built using LangChain, LangGraph, Ollama, and FAISS. It retrieves relevant information from uploaded documents and generates answers using a locally running language model.

## Features

- Supports document ingestion and text chunking.
- Generates embeddings locally using Ollama.
- Stores and retrieves document chunks using FAISS.
- Uses LangGraph to manage the question-answering workflow.
- Grades retrieved documents for relevance.
- Rewrites questions when retrieval needs improvement.
- Generates answers using retrieved context.
- Evaluates responses against a 20-question dataset.
- Includes pytest unit tests.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| LangChain | Document processing and LLM integration |
| LangGraph | Question-answering workflow |
| Ollama | Runs local language and embedding models |
| `llama3.2` | Answer generation and workflow decisions |
| `nomic-embed-text` | Text embeddings |
| FAISS | Vector storage and similarity retrieval |
| Streamlit | Chat interface |
| Pytest | Unit testing |

## Architecture

### Document ingestion

1. Documents are placed in `data/documents/`.
2. The ingestion pipeline loads supported files.
3. Documents are split into smaller text chunks.
4. Ollama generates embeddings for the chunks.
5. FAISS stores the vectors for retrieval.

### Question-answering workflow

1. The user submits a question.
2. The workflow contextualizes follow-up questions.
3. The router classifies the request.
4. FAISS retrieves relevant document chunks.
5. A relevance grader filters retrieved chunks.
6. If needed, the workflow rewrites the question and retries retrieval.
7. The LLM generates an answer using the available context.
8. The response is returned with source citations.

## Project Structure

```text
Document Chatbot/
├── data/
│   └── documents/
├── src/
│   ├── ingestion/
│   │   └── ingest.py
│   ├── retrieval/
│   │   └── retriever.py
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   └── builder.py
│   └── evaluation/
│       └── evaluate.py
├── tests/
│   └── test_evaluation.py
├── evaluation_questions.json
├── evaluation_results.json
├── requirements.txt
├── .env
└── README.md
```

## Prerequisites

- Python 3.13
- Ollama installed and running
- Git (optional)

Download Ollama from https://ollama.com/.

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
python -m pip install pytest faiss-cpu
```

### 3. Download the Ollama models

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```text
OLLAMA_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
```

Keep your environment-specific settings out of version control.

## Running the Application

### 1. Add documents

Place supported documents in:

```text
data/documents/
```

### 2. Ingest documents

Run from the project root:

```powershell
python -m src.ingestion.ingest
```

### 3. Start the chatbot

If the Streamlit entry point is `app.py`, run:

```powershell
streamlit run app.py
```

If your Streamlit entry point has a different filename, use that filename instead.

## Evaluation

The evaluation dataset contains 20 questions with expected answers and source documents.

Run the evaluation:

```powershell
python -m src.evaluation.evaluate
```

The script writes its results to `evaluation_results.json`, including the actual answer, expected and retrieved sources, answer-check result, latency, and pass/fail status.

### Current results

| Metric | Result |
|---|---:|
| Evaluation questions | 20 |
| Passed | 14 |
| Failed | 6 |
| Pass rate | 70% |
| Target | At least 15/20 (75%) |

These results reflect the last confirmed evaluation run and may change when the code or evaluation logic changes.

## Unit Tests

Run the tests from the project root:

```powershell
python -m pytest -v
```

Current result:

- 5 tests passed.
- 0 test failures.

The tests cover text normalization and evaluation dataset loading and validation.

## Design Choices

- **Local inference:** Ollama runs the language model locally, avoiding a dependency on a hosted LLM API.
- **Local embeddings:** `nomic-embed-text` creates embeddings for document chunks.
- **FAISS:** Used for vector retrieval after encountering issues with the earlier Chroma setup.
- **LangGraph:** Separates routing, retrieval, relevance grading, query rewriting, generation, and fallback behavior into workflow nodes.
- **Source-grounded answers:** The generation prompt instructs the model to use supplied context and cite sources.
- **Automated evaluation:** A fixed question dataset makes it easier to compare changes across runs.

## Limitations

- The current evaluation score is 70%, below the 75% target.
- Some answers may be marked incorrectly by the evaluation method, particularly when the wording differs from the expected answer.
- Retrieval and generation can miss relevant facts even when they exist in the source documents.
- Source retrieval alone does not guarantee that the generated answer is correct or fully grounded.
- Only the file types supported by the ingestion pipeline can be processed.
- The application depends on the local Ollama service and the required models being available.

## Future Improvements

- Improve retrieval for questions requiring multiple facts.
- Improve routing for document-related questions.
- Strengthen answer grounding and citation verification.
- Expand unit tests to cover graph nodes, retrieval, and fallback behavior.
- Improve evaluation with semantic correctness and faithfulness checks.
- Test with a larger collection of documents and questions.

## License

Add the license appropriate for your project before publishing or distributing it.
