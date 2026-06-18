from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from modsa_rag.config import Settings
from modsa_rag.ingest import get_vector_store


SYSTEM_PROMPT = """You are MOD-SA, a KMUTT Student Affairs RAG assistant.

Use only the retrieved context to answer. If the context does not contain enough
information, say that you do not have enough verified information and recommend
contacting the relevant KMUTT office.

Answer in Thai when the question is Thai. Answer in English when the question is
English. Be concise, accurate, and careful with dates, rules, eligibility,
deadlines, fees, scholarships, and registration details.

Retrieved context:
{context}
"""


def build_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.resolved_llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
    )


def format_context(docs) -> str:
    parts: list[str] = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page")
        label = f"{source}"
        if page is not None:
            label = f"{label}, page {int(page) + 1}"
        parts.append(f"[{index}] {label}\n{doc.page_content}")
    return "\n\n".join(parts)


def source_summary(docs) -> list[dict[str, object]]:
    seen: set[tuple[object, object]] = set()
    sources: list[dict[str, object]] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        item: dict[str, object] = {"source": source}
        if page is not None:
            item["page"] = int(page) + 1
        sources.append(item)
    return sources


def answer_question(settings: Settings, question: str) -> dict[str, object]:
    vector_store = get_vector_store(settings)
    retriever = vector_store.as_retriever(search_kwargs={"k": settings.retrieval_k})
    docs = retriever.invoke(question)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    messages = prompt.format_messages(
        context=format_context(docs),
        question=question,
    )
    response = build_llm(settings).invoke(messages)

    return {
        "answer": response.content,
        "sources": source_summary(docs),
    }
