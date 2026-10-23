
from typing import Literal

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

from src.config import (
    OLLAMA_MODEL,
    LLM_PROVIDER,
    GROQ_MODEL,
    GROQ_API_KEY,
)
from src.retrieval.retriever import get_retriever


# Initialize the selected LLM provider.
if LLM_PROVIDER == "groq":
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing from the .env file."
        )

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )

elif LLM_PROVIDER == "ollama":
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )

else:
    raise ValueError(
        f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. "
        "Choose 'ollama' or 'groq'."
    )


# Structured output schemas.
class RouteDecision(BaseModel):
    route: Literal[
        "greeting",
        "document_question",
        "out_of_scope",
    ] = Field(description="The correct route for the user's request")


class RelevanceDecision(BaseModel):
    relevant: Literal["yes", "no"] = Field(
        description="Whether the document helps answer the question"
    )


# Reuse the existing FAISS retriever.
retriever = get_retriever()


def route_question(state: dict) -> dict:
    question = state.get("rewritten_question") or state["question"]

    router = llm.with_structured_output(RouteDecision)

    decision = router.invoke([
        (
            "system",
            """
            You are a request classifier for a document chatbot.

            Choose exactly one route:

            1. greeting
               Use ONLY for greetings or simple social conversation.
               Examples: "Hi", "Hello", "How are you?"

            2. document_question
               Use for ANY question asking for factual information,
               including questions that may or may not be answered
               by the documents.
               Examples:
               - "How many days of annual paid leave do employees receive?"
               - "What is the company's leave policy?"
               - "Who is the CEO of the company?"
               - "Summarize the uploaded PDF."

            3. out_of_scope
               Use ONLY for requests unrelated to document question
               answering.
               Examples:
               - "Write a Python game."
               - "Give me a recipe for pizza."

            IMPORTANT:
            A question about a company's policy is a document_question.
            A factual question must NEVER be classified as out_of_scope
            just because you do not know its answer.

            Return only the structured classification.
            """
        ),
        ("human", question),
    ])

    print(f"Route: {decision.route}", flush=True)

    return {"route": decision.route}


def retrieve(state: dict) -> dict:
    """Retrieve relevant candidate chunks from FAISS."""
    question = (
        state.get("rewritten_question")
        or state["question"]
    )

    k = state.get("retrieval_k")
    current_retriever = get_retriever(k=k)
    documents = current_retriever.invoke(question)

    print(f"Retrieved {len(documents)} chunks.", flush=True)

    return {
        "documents": documents,
        "rewritten_question": question,
    }


def grade_documents(state: dict) -> dict:
    """Keep only chunks that help answer the question."""
    question = (
        state.get("rewritten_question")
        or state["question"]
    )

    grader = llm.with_structured_output(RelevanceDecision)
    relevant_documents = []

    for doc in state.get("documents", []):
        decision = grader.invoke([
            (
                "system",
                """
                Decide whether the document contains information
                useful for answering the question.

                Answer yes if it contains relevant facts or
                information that helps answer the question.
                Answer no if it is unrelated.

                Do not judge whether the document contains the
                complete answer; judge relevance only.
                """
            ),
            (
                "human",
                f"Question: {question}\n\nDocument:\n{doc.page_content}",
            ),
        ])

        if decision.relevant == "yes":
            relevant_documents.append(doc)

    print(
        f"Relevant chunks: {len(relevant_documents)} "
        f"of {len(state.get('documents', []))}",
        flush=True,
    )

    return {"documents": relevant_documents}


def rewrite_query(state: dict) -> dict:
    """Rewrite the question to improve retrieval."""
    previous_query = (
        state.get("rewritten_question")
        or state["question"]
    )

    response = llm.invoke([
        (
            "system",
            """
            Rewrite the user's question to improve document search.
            Keep the original meaning. Use clear keywords.
            Return only the rewritten question, with no explanation.
            """
        ),
        ("human", previous_query),
    ])

    rewritten = response.content.strip()
    retry_count = state.get("retry_count", 0) + 1

    print(f"Rewritten question: {rewritten}", flush=True)
    print(f"Retry count: {retry_count}", flush=True)

    return {
        "rewritten_question": rewritten,
        "retry_count": retry_count,
    }


def generate(state: dict) -> dict:
    """Generate an answer grounded in retrieved document context."""
    question = (
        state.get("rewritten_question")
        or state["question"]
    )

    documents = state.get("documents", [])
    context_parts = []

    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")

        if isinstance(page, int):
            citation = f"{source}, p. {page + 1}"
        else:
            citation = source

        context_parts.append(
            f"[Source: {citation}]\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    response = llm.invoke([
        (
            "system",
            """
            Answer the question using only the supplied context.

            Do not invent facts.
            Cite factual claims using the source labels provided.
            For example: [Sample.pdf, p. 3].
            For text files without page numbers, cite the filename.
            If the context does not contain the answer, say:
            "I couldn't find this in the provided documents."
            """
        ),
        (
            "human",
            f"Context:\n{context}\n\nQuestion:\n{question}",
        ),
    ])

    return {"generation": response.content.strip()}


def fallback(state: dict) -> dict:
    """Return a safe response when the graph cannot answer."""
    route = state.get("route", "document_question")

    if route == "greeting":
        answer = (
            "Hello! I'm your document assistant. "
            "Ask me a question about the provided documents."
        )
    elif route == "out_of_scope":
        answer = (
            "I can help answer questions about the provided "
            "documents. Please ask a document-related question."
        )
    else:
        answer = "I couldn't find this in the provided documents."

    return {"generation": answer}


def contextualize_question(state: dict) -> dict:
    """Rewrite a follow-up using the previous conversation."""
    question = state["question"]
    history = state.get("messages", [])

    if not history:
        return {"rewritten_question": question}

    history_text = "\n".join(
        f"{message.type}: {message.content}"
        for message in history
    )

    response = llm.invoke([
        (
            "system",
            """
            Rewrite the latest user question as a standalone question
            using the conversation history.

            Resolve references such as "it", "that", and "its".
            Preserve the user's original meaning.
            If the question is already clear, keep its meaning unchanged.
            Return only the rewritten question.
            """
        ),
        (
            "human",
            f"Conversation history:\n{history_text}\n\n"
            f"Latest question:\n{question}",
        ),
    ])

    standalone_question = response.content.strip()
    print(
        f"Contextualized question: {standalone_question}",
        flush=True,
    )

    return {"rewritten_question": standalone_question}


def record_turn(state: dict) -> dict:
    """Save this question and answer to the conversation history."""
    return {
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=state["generation"]),
        ]
    }
