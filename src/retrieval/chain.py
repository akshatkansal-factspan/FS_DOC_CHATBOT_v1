
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

from src.config import OLLAMA_MODEL
from src.retrieval.retriever import get_retriever


PROMPT_TEMPLATE = """
You are a document question-answering assistant.

Answer using ONLY the provided context.
Do not invent facts.

Cite the source using [source, p. N] when a page number is available.
If the answer is not present, say:
"I couldn't find this in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""


def format_documents(documents):
    formatted = []

    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")

        if isinstance(page, int):
            page_label = f"p. {page + 1}"
        else:
            page_label = "page unavailable"

        formatted.append(
            f"[Source: {source}, {page_label}]\n{doc.page_content}"
        )

    return "\n\n".join(formatted)


def build_chain():
    retriever = get_retriever()

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def answer_question(inputs):
        question = inputs["question"]

        # Pass only a string to the retriever.
        documents = retriever.invoke(question)
        context = format_documents(documents)

        messages = prompt.invoke({
            "context": context,
            "question": question,
        })

        return (llm | StrOutputParser()).invoke(messages)

    return RunnableLambda(answer_question)


if __name__ == "__main__":
    chain = build_chain()

    question = "Who is the CEO of the company?"
    answer = chain.invoke({"question": question})

    print(f"Question: {question}\n")
    print(f"Answer: {answer}")
